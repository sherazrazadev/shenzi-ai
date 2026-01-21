# Agent Service

This AI-powered service provides conversational AI capabilities using LangChain and LangGraph with GROQ API.

## Features

- **LangGraph Agent**: Built with LangGraph for structured conversation flow
- **GROQ Integration**: Uses GROQ API for fast LLM inference
- **Streaming Responses**: Real-time streaming of AI responses
- **Context Awareness**: Maintains conversation context across messages

## Endpoints

- `GET /`: Health check
- `POST /chat`: Regular chat endpoint (non-streaming)
- `POST /chat/stream`: Streaming chat endpoint for real-time responses
- `GET /tasks`: Get user tasks (placeholder)
- `POST /tasks`: Create new task (placeholder)

## Environment Variables

- `GROQ_API_KEY`: Your GROQ API key (required)

## Dependencies

- langchain
- langchain-groq
- langgraph
- fastapi
- uvicorn