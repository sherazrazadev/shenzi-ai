# Agent Service

This AI-powered service provides conversational AI capabilities using LangChain and LangGraph with GROQ API.

## Features

- **LangGraph Agent**: Built with LangGraph for structured conversation flow
- **GROQ Integration**: Uses GROQ API for fast LLM inference
- **Ollama Option**: Optionally run a local Ollama model (set `AGENT_MODEL` and `OLLAMA_BASE_URL`)
- **Streaming Responses**: Real-time streaming of AI responses
- **Context Awareness**: Maintains conversation context across messages

## Endpoints

- `GET /`: Health check
- `POST /chat`: Regular chat endpoint (non-streaming)
- `POST /chat/stream`: Streaming chat endpoint for real-time responses
- `GET /tasks`: Get user tasks (placeholder)
- `POST /tasks`: Create new task (placeholder)

## Environment Variables

- `GROQ_API_KEY`: Your GROQ API key (required if using GROQ)
- `TAVILY_API_KEY`: Your Tavily API key (required)
- `OLLAMA_BASE_URL`: Optional Ollama base URL (defaults to `http://localhost:11434`)
- `AGENT_MODEL`: Optional model name for Ollama (defaults to `qwen2.5-coder:7b`)

**Docker note**: To run the Ollama-backed service in Docker set `APP_MODULE=app.main_ollama` (or set the env in your `docker-compose.yml`).

## Dependencies

- langchain
- langchain-groq
- langgraph
- fastapi
- uvicorn
