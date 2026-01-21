# Shenzi Microservices Project

This is a microservices-based project built with Python, FastAPI, Next.js, and MongoDB. The project includes services for authentication, verification, an AI agent for daily tasks, a user interface, and a database backend.

## Services

- **API Gateway**: Routes requests to appropriate services, handles load balancing and authentication.
- **Auth Service**: Handles user authentication and JWT token management.
- **Agent Service**: AI-powered service using LangChain and LangGraph with GROQ API for conversational AI and task assistance.
- **Verification Service**: Handles email verification and other validations.
- **Database Service**: MongoDB backend service for data operations.
- **Client UI Service**: Next.js-based user interface with chat interface for AI assistant.
- **MongoDB**: Database service.

## API Endpoints (via Gateway)

- `/auth/*` - Authentication endpoints
- `/call-ai/chat` - AI chat (regular)
- `/call-ai/chat/stream` - AI chat (streaming)
- `/api/*` - Database operations
- `/verify/*` - Verification endpoints

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Services

1. Clone the repository.
2. Navigate to the project root.
3. Run `docker-compose up --build` to start all services.

The API Gateway will be available at `http://localhost:8000`, and the UI at `http://localhost:3000`.

### Individual Services

- API Gateway: `http://localhost:8000`
- Auth: `http://localhost:8001`
- Agent: `http://localhost:8002`
- Verification: `http://localhost:8003`
- Database: `http://localhost:8004`
- UI: `http://localhost:3000`
- MongoDB: `localhost:27017`

## Development

Each service is in its own directory under `services/`. To develop a service:

1. Navigate to the service directory.
2. For Python services: Install dependencies with `pip install -r requirements.txt` and run with `uvicorn app.main:app --reload`
3. For UI service: Install dependencies with `npm install` and run with `npm run dev`

## Testing

Tests are located in the `tests/` directory of each service.