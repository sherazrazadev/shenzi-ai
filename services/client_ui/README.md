# Client UI

This is the user interface for the Shenzi AI Assistant, built with Next.js and TypeScript.

## Features

- **Chat Interface**: Clean, responsive chat UI for interacting with the AI assistant
- **Real-time Messaging**: Send and receive messages with typing indicators
- **Component-based Architecture**: Modular components for maintainability
- **API Integration**: Connects to the API gateway for backend services

## Components

- `ChatContainer`: Main chat interface managing state and messages
- `ChatMessage`: Individual message display component
- `ChatInput`: Message input form with send functionality

## API Endpoints

- `POST /api/chat`: Send chat messages (proxies to API gateway)

## Getting Started

1. Install dependencies: `npm install`
2. Run development server: `npm run dev`
3. Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

- `API_GATEWAY_URL`: URL of the API gateway (default: http://localhost:8000)

## Docker

The service is containerized and runs on port 3000 in the Docker Compose setup.
