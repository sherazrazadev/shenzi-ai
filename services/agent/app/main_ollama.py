from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.tools import Tool
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, TypedDict
import os
import asyncio
from dotenv import load_dotenv

# Import system prompt from constants
from app.constants.prompt import SYSTEM_PROMPT


# Import custom tool(s)
from app.tools.search_tavily import get_tavily_tool

from app.tools.sheet_tasks import get_tasks_for_date
from app.tools.logger import BackgroundLogger
logger = BackgroundLogger("agent_ollama.log")

# Load environment variables
load_dotenv()

app = FastAPI(title="Agent Service (Ollama)", tags="AI Service", description="AI Agent service using LangGraph and Ollama")

# Initialize Ollama LLM
# Optional: set OLLAMA_BASE_URL (defaults to http://localhost:11434) and AGENT_MODEL (defaults to qwen2.5-coder:7b)
ollama_base = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
agent_model = os.getenv("AGENT_MODEL", "qwen2.5-coder:7b")

llm = ChatOllama(
    model=agent_model,
    base_url=ollama_base,
    temperature=0.7,
    num_predict=400
)

# Initialize Tavily search tool (moved to tools/search_tavily.py)
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required")



# Pydantic schema for Tavily tool
class TavilySearchInput(BaseModel):
    query: str
    start_date: str = None
    end_date: str = None

def tavily_search_v2_tool(input: TavilySearchInput):
    args = input.dict(exclude_none=True)
    start_date = args.get('start_date')
    end_date = args.get('end_date')
    if start_date and end_date and start_date == end_date:
        args.pop('end_date')
    return get_tavily_tool(api_key=tavily_api_key, max_results=3).invoke(args)

search_tool = Tool(
    name="tavily_search_v2",
    description="Search the web for up-to-date information. Use for current events, news, weather, etc.",
    args_schema=TavilySearchInput,
    func=tavily_search_v2_tool,
)


# Pydantic schema for sheet_tasks tool
class SheetTasksInput(BaseModel):
    date: str

def sheet_tasks_tool(input: SheetTasksInput):
    """Get user's tasks for a specific date from Google Sheet."""
    date = input.date
    tasks = get_tasks_for_date(date)
    if not tasks:
        return f"No tasks found for {date}."
    formatted = [
        f"- {t.get('Time_Slot', '')}: {t.get('Task_Name', '')} [{t.get('Category', '')}] (Priority: {t.get('Priority', '')}, Status: {t.get('Status', '')}, Habit: {t.get('Habit_Check', '')}) Notes: {t.get('Notes', '')}"
        for t in tasks
    ]
    return f"Tasks for {date}:\n" + "\n".join(formatted)

from langchain_core.tools import Tool

sheet_tasks_langchain_tool = Tool(
    name="sheet_tasks",
    description=(
        "Use this tool to get the user's personal daily tasks, habits, and schedule from their Google Sheet. "
        "Call this tool whenever the user asks about their tasks, schedule, or habits for a specific date. "
        "Input: date as YYYY-MM-DD. Output: formatted list of tasks."
    ),
    args_schema=SheetTasksInput,
    func=sheet_tasks_tool,
)


# Register all tools
tools = [
    search_tool,
    sheet_tasks_langchain_tool,
]

# Bind tools to LLM (if supported)
try:
    llm_with_tools = llm.bind_tools(tools)
except Exception:
    # Fallback to plain llm if bind_tools is not supported by the Ollama wrapper
    llm_with_tools = llm

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: bool

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

import json, re

def parse_tool_call_from_text(text: str):
    """Try to parse a tool call JSON object from the model's text output.
    Returns dict with keys 'name' and 'args' or None if not found."""
    if not text:
        return None
    text = re.sub(r"```.*?```", lambda m: m.group(0).replace('```', ''), text, flags=re.S)
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            if 'tool' in parsed and 'args' in parsed:
                return {'name': parsed['tool'], 'args': parsed['args']}
            if 'tool_call' in parsed and isinstance(parsed['tool_call'], dict):
                tc = parsed['tool_call']
                if 'name' in tc and 'args' in tc:
                    return {'name': tc['name'], 'args': tc['args']}
    except Exception:
        return None
    return None

def is_time_sensitive_query(text: str) -> bool:
    """Return True if the text likely requires up-to-date info or a web search (news, holidays, current events)."""
    if not text:
        return False
    t = text.lower()
    keywords = ['today', 'news', 'latest', 'current', 'now', 'holiday', 'holidays', 'weekend']
    if any(k in t for k in keywords):
        return True
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','sept','oct','nov','dec']
    for m in months:
        if m in t:
            return True
    if any(char.isdigit() for char in t):
        if any(m in t for m in months):
            return True
    return False


def is_sheet_tasks_query(text: str) -> bool:
    """Return True if the text is asking about tasks in the user's sheet or Excel."""
    if not text:
        return False
    t = text.lower()
    keywords = ['task', 'tasks', 'sheet', 'excel', 'google sheet', 'my sheet', 'my excel', 'todo', 'to-do']
    return any(k in t for k in keywords)


def extract_date_from_text(text: str):
    """Try simple date extraction from text. Returns YYYY-MM-DD or None.
    Supports YYYY-MM-DD and patterns like 'feb 6' (assumes current year)."""
    if not text:
        return None
    import re
    from datetime import date
    # YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    # Month name + day (e.g., feb 6 or feb 6th)
    months = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
    }
    m2 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b\s*(\d{1,2})", text.lower())
    if m2:
        mon = months.get(m2.group(1)[:3])
        day = int(m2.group(2))
        today = date.today()
        year = today.year
        # naive: if month already passed and it's not the same year, keep current year
        try:
            from datetime import datetime
            dt = datetime(year, mon, day)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None
    return None

# Define the agent function
def agent(state: State):
    messages = state["messages"]
    logger.info(f"Agent received messages: {messages}")
    try:
        # Proactive detection for time-sensitive queries
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        # If this looks like a sheet/tasks query, proactively call sheet_tasks (prefer explicit date extraction)
        if last_user_msg and is_sheet_tasks_query(last_user_msg):
            logger.info(f"Detected sheet-tasks query: {last_user_msg}")
            try:
                date_arg = extract_date_from_text(last_user_msg)
                if not date_arg:
                    # default to today and ask the user if they meant another date
                    from datetime import date as _d
                    date_arg = _d.today().isoformat()
                    note = f"(Defaulting to today's date: {date_arg}. If you'd like a different date, please specify.)"
                else:
                    note = ''

                print(f"[DEBUG] Proactively calling sheet_tasks for date: {date_arg}")
                result = sheet_tasks_tool(SheetTasksInput(date=date_arg))
                logger.info(f"Proactive sheet_tasks result: {result}")
                # Build messages for model to synthesize a friendly answer
                summary_text = f"Tasks for {date_arg}:\n{result}\n{note}" if result else f"No tasks found for {date_arg}. {note}"
                instruction = (
                    "You have access to the user's sheet tasks (shown above)."
                    " Summarize them for the user in a friendly, concise way."
                )
                messages_with_result = messages + [HumanMessage(content=summary_text), HumanMessage(content=instruction)]
                try:
                    msgs_for_debug = [f"{type(m).__name__}: {getattr(m, 'content', str(m))}" for m in messages_with_result]
                    logger.debug(f"Messages sent to model (proactive sheet): {msgs_for_debug}")
                    print(f"[DEBUG] Messages sent to model (proactive sheet): {msgs_for_debug}")
                except Exception:
                    pass
                final_response = llm.invoke(messages_with_result)
                logger.info(f"Final response after proactive sheet_tasks: {final_response}")
                resp_text = getattr(final_response, 'content', '') if final_response else str(final_response)
                if resp_text and ("don't" in resp_text.lower() and ("real" in resp_text.lower() or "access" in resp_text.lower() or "internet" in resp_text.lower())):
                    logger.info("Model refused to use sheet results; returning the sheet content directly.")
                    from langchain_core.messages import AIMessage as _AIMessage
                    return {"messages": [_AIMessage(content=summary_text)], "tools_used": True}
                return {"messages": [final_response], "tools_used": True}
            except Exception as e:
                logger.error(f"Proactive sheet_tasks failed: {e}")

        if last_user_msg and is_time_sensitive_query(last_user_msg):
            logger.info(f"Detected time-sensitive query: {last_user_msg}")
            search_args = {"query": last_user_msg}
            try:
                search_result = get_tavily_tool(api_key=tavily_api_key, max_results=3).invoke(search_args)
                logger.info(f"Proactive tavily_search result: {search_result}")
                print(f"[DEBUG] Proactively called tavily_search with args: {search_args}\nResult: {search_result}")

                # Format search results into a concise, model-friendly human message
                def format_search_result(sr: dict) -> str:
                    parts = []
                    parts.append(f"Search query: {sr.get('query')}")
                    results = sr.get('results', [])
                    if not results:
                        parts.append("No results found.")
                    else:
                        for i, r in enumerate(results[:5], start=1):
                            title = r.get('title') or ''
                            url = r.get('url') or ''
                            snippet = (r.get('content') or r.get('snippet') or '')[:300]
                            parts.append(f"{i}. {title} — {url}\n   {snippet}")
                    return "\n".join(parts)

                summary_text = format_search_result(search_result)
                instruction = (
                    "You have access to recent search results (shown above)."
                    " Use these results to answer the user's question directly and do NOT state that you lack real-time access."
                    " Answer concisely and cite the relevant source URLs."
                )
                messages_with_result = messages + [HumanMessage(content=summary_text), HumanMessage(content=instruction)]
                # Debug: log the messages we will send to the model
                try:
                    msgs_for_debug = [f"{type(m).__name__}: {getattr(m, 'content', str(m))}" for m in messages_with_result]
                    logger.debug(f"Messages sent to model (proactive): {msgs_for_debug}")
                    print(f"[DEBUG] Messages sent to model (proactive): {msgs_for_debug}")
                except Exception:
                    pass
                final_response = llm.invoke(messages_with_result)
                logger.info(f"Final response after proactive search: {final_response}")
                resp_text = getattr(final_response, 'content', '') if final_response else str(final_response)
                # If model still refuses to use external info, fall back to auto summary
                if resp_text and ("don't" in resp_text.lower() and ("real" in resp_text.lower() or "access" in resp_text.lower() or "internet" in resp_text.lower())):
                    logger.info("Model refused to use external results; returning auto-generated summary from search results.")
                    auto_summary = "Here are the latest results I found:\n" + summary_text
                    from langchain_core.messages import AIMessage as _AIMessage
                    return {"messages": [_AIMessage(content=auto_summary)], "tools_used": True}
                return {"messages": [final_response], "tools_used": True}
            except Exception as e:
                logger.error(f"Proactive search failed: {e}")

        ai_message = llm_with_tools.invoke(messages)
        logger.info(f"AI message: {ai_message}")

        # Check for tool calls
        tool_calls = []
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            tool_calls = ai_message.tool_calls
        else:
            parsed = parse_tool_call_from_text(getattr(ai_message, 'content', '') if hasattr(ai_message, 'content') else str(ai_message))
            if parsed:
                parsed['id'] = 'parsed_1'
                tool_calls = [parsed]

        if tool_calls:
            logger.info(f"Tool calls detected: {tool_calls}")
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                logger.info(f"Tool call: {tool_name}, args: {tool_args}")
                if tool_name == 'tavily_search' or tool_name == 'tavily_search_v2':
                    result = search_tool.invoke(tool_args)
                    logger.info(f"Tavily search result: {result}")
                    print(f"[DEBUG] Called tavily_search with args: {tool_args}\nResult: {result}")
                    # Format the search result into a clear human-readable summary for Ollama
                    def format_search_result(sr: dict) -> str:
                        parts = []
                        parts.append(f"Search query: {sr.get('query')}")
                        results = sr.get('results', [])
                        if not results:
                            parts.append("No results found.")
                        else:
                            for i, r in enumerate(results[:5], start=1):
                                title = r.get('title') or ''
                                url = r.get('url') or ''
                                snippet = (r.get('content') or r.get('snippet') or '')[:300]
                                parts.append(f"{i}. {title} — {url}\n   {snippet}")
                        return "\n".join(parts)

                    summary_text = format_search_result(result)
                    tool_results.append(HumanMessage(content=summary_text))
                elif tool_name == 'sheet_tasks':
                    logger.info(f"Calling sheet_tasks_tool with input: {tool_args}")
                    print(f"[DEBUG] Calling sheet_tasks_tool with input: {tool_args}")
                    result = sheet_tasks_tool(SheetTasksInput(**tool_args))
                    logger.info(f"Sheet tasks result: {result}")
                    print(f"[DEBUG] sheet_tasks_tool result: {result}")
                    tool_results.append(HumanMessage(content=str(result)))
            # Synthesize a user-focused answer using both LLM and tool results
            synthesis_prompt = (
                "You used a tool and got these results:\n"
                f"{getattr(tool_results[0], 'content', '') if tool_results else ''}\n"
                "Now, using both your own knowledge and these tool results, write a clear, helpful, and user-focused answer."
            )
            logger.info(f"Synthesis prompt: {synthesis_prompt}")
            messages = messages + [ai_message] + tool_results + [HumanMessage(content=synthesis_prompt + " Do not say you cannot access real-time information; use the results above directly.")]
            try:
                msgs_for_debug = [f"{type(m).__name__}: {getattr(m, 'content', str(m))}" for m in messages]
                logger.debug(f"Messages sent to model (tool-synthesis): {msgs_for_debug}")
                print(f"[DEBUG] Messages sent to model (tool-synthesis): {msgs_for_debug}")
            except Exception:
                pass
            final_response = llm.invoke(messages)
            logger.info(f"Final response: {final_response}")
            resp_text = getattr(final_response, 'content', '') if final_response else str(final_response)
            if resp_text and ("don't" in resp_text.lower() and ("real" in resp_text.lower() or "access" in resp_text.lower() or "internet" in resp_text.lower())):
                logger.info("Model refused to use tool results; returning auto-generated summary from tool results.")
                # build auto summary from tool_results contents
                summary_parts = []
                for tr in tool_results:
                    summary_parts.append(getattr(tr, 'content', str(tr)))
                auto_summary = "Here are the results I found:\n" + "\n\n".join(summary_parts)
                from langchain_core.messages import AIMessage as _AIMessage
                return {"messages": [_AIMessage(content=auto_summary)], "tools_used": True}
            return {"messages": [final_response], "tools_used": True}
        else:
            logger.info("No tool calls detected. Returning AI message.")
            return {"messages": [ai_message], "tools_used": False}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        print(f"[ERROR] Agent error: {e}")
        # If tool calling fails, fall back to direct response without tools
        try:
            ai_message = llm.invoke(messages)
            return {"messages": [ai_message], "tools_used": False}
        except Exception as e2:
            logger.error(f"Fallback error: {e2}")
            print(f"[ERROR] Fallback error: {e2}")
            error_message = AIMessage(content=f"Sorry, I encountered an error: {str(e2)}")
            return {"messages": [error_message], "tools_used": False}

# Create the graph
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Compile the graph
app_graph = graph.compile()

class ChatRequest(BaseModel):
    messages: list[dict]

class StreamRequest(BaseModel):
    messages: list[dict]

@app.get("/")
async def root():
    return {"message": "Agent Service (Ollama) is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        # Parse messages
        parsed_messages = []
        for msg in request.messages:
            if msg["role"] == "user":
                parsed_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                parsed_messages.append(AIMessage(content=msg["content"]))

        # Create initial state with conversation history
        initial_state = {"messages": parsed_messages, "tools_used": False}

        # Run the graph
        result = app_graph.invoke(initial_state)

        # Get the last message (AI response)
        ai_response = result["messages"][-1].content

        # If no tools were used, add the note
        if not result.get("tools_used", False):
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M:%S %p")
            ai_response += f"\n\n{current_time}\n\nNo function was called. To continue, you can make a call to a function to get information about a specific topic."

        return {"response": ai_response}
    except Exception as e:
        return {"error": f"Failed to process chat: {str(e)}"}

@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """Streaming chat endpoint"""
    async def generate():
        try:
            # Parse messages
            parsed_messages = []
            for msg in request.messages:
                if msg["role"] == "user":
                    parsed_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    parsed_messages.append(AIMessage(content=msg["content"]))

            # Create initial state with conversation history
            initial_state = {"messages": parsed_messages}

            # Run the graph (non-streaming for now, to handle tools)
            result = app_graph.invoke(initial_state)

            # Get the last message (AI response)
            ai_response = result["messages"][-1].content

            # Stream the response character by character
            for char in ai_response:
                yield f"data: {char}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/tasks")
async def get_tasks():
    """Get user's tasks - placeholder for now"""
    return {"tasks": []}

@app.post("/tasks")
async def create_task(task: dict):
    """Create a new task - placeholder for now"""
    return {"message": "Task created successfully", "task": task, "note": "Full task management integration coming soon"}
