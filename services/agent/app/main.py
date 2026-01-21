from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, TypedDict
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Agent Service", description="AI Agent service using LangGraph and GROQ")

# Initialize GROQ LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=groq_api_key,
    temperature=0.7,
    max_tokens=1024
)

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Shenzi, a helpful AI assistant for daily task management and personal productivity.

You can help users with:
- Task management (creating, listing, updating tasks)
- Setting reminders and schedules
- Answering questions
- Providing productivity tips
- Weather information (when available)
- General assistance

Be friendly, helpful, and concise in your responses. If you need to access external data or perform actions, let the user know what you're doing.

Always maintain context from previous messages in the conversation."""),
    ("placeholder", "{messages}")
])

# Define the agent function
def agent(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create the graph
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Compile the graph
app_graph = graph.compile()

class ChatRequest(BaseModel):
    message: str

class StreamRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Agent Service is running with LangGraph and GROQ"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        # Create initial state with user message
        initial_state = {"messages": [HumanMessage(content=request.message)]}

        # Run the graph
        result = app_graph.invoke(initial_state)

        # Get the last message (AI response)
        ai_response = result["messages"][-1].content

        return {"response": ai_response}
    except Exception as e:
        return {"error": f"Failed to process chat: {str(e)}"}

@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """Streaming chat endpoint"""
    async def generate():
        try:
            # Create initial state with user message
            initial_state = {"messages": [HumanMessage(content=request.message)]}

            # Run the graph with streaming
            async for event in app_graph.astream(initial_state):
                for node_name, messages in event.items():
                    if node_name == "agent" and messages.get("messages"):
                        for message in messages["messages"]:
                            if isinstance(message, AIMessage):
                                # Stream the response token by token
                                content = message.content
                                for char in content:
                                    yield f"data: {char}\n\n"
                                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                                yield "data: [DONE]\n\n"
                                return
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
    return {"tasks": [], "message": "Task management integration coming soon"}

@app.post("/tasks")
async def create_task(task: dict):
    """Create a new task - placeholder for now"""
    return {"message": "Task created successfully", "task": task, "note": "Full task management integration coming soon"}