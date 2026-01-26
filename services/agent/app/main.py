from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
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
logger = BackgroundLogger("agent.log")

# Load environment variables
load_dotenv()

app = FastAPI(title="Agent Service", tags="AI Service", description="AI Agent service using LangGraph and GROQ")

# Initialize GROQ LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.7,
    max_tokens=400  # Reduced for shorter responses
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

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tools_used: bool

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

# Define the agent function
def agent(state: State):
    messages = state["messages"]
    logger.info(f"Agent received messages: {messages}")
    try:
        ai_message = llm_with_tools.invoke(messages)
        logger.info(f"AI message: {ai_message}")

        # Check for tool calls
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            logger.info(f"Tool calls detected: {ai_message.tool_calls}")
            tool_results = []
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                logger.info(f"Tool call: {tool_name}, args: {tool_args}")
                if tool_name == 'tavily_search':
                    result = search_tool.invoke(tool_args)
                    logger.info(f"Tavily search result: {result}")
                    print(f"[DEBUG] Called tavily_search with args: {tool_args}\nResult: {result}")
                    tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
                elif tool_name == 'sheet_tasks':
                    logger.info(f"Calling sheet_tasks_tool with input: {tool_args}")
                    print(f"[DEBUG] Calling sheet_tasks_tool with input: {tool_args}")
                    result = sheet_tasks_tool(SheetTasksInput(**tool_args))
                    logger.info(f"Sheet tasks result: {result}")
                    print(f"[DEBUG] sheet_tasks_tool result: {result}")
                    tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
            # Synthesize a user-focused answer using both LLM and tool results
            synthesis_prompt = (
                "You used a tool and got these results:\n"
                f"{tool_results[0].content if tool_results else ''}\n"
                "Now, using both your own knowledge and these tool results, write a clear, helpful, and user-focused answer."
            )
            logger.info(f"Synthesis prompt: {synthesis_prompt}")
            messages = messages + [ai_message] + tool_results + [HumanMessage(content=synthesis_prompt)]
            final_response = llm.invoke(messages)
            logger.info(f"Final response: {final_response}")
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
    return {"message": "Agent Service is running"}

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