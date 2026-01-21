# API Gateway

This service acts as the entry point for all API requests, routing them to appropriate services.

## Routing

Requests are routed based on the path prefix:
- `/auth*` -> Auth Service
- `/call-ai*` -> Agent Service (AI)
- `/api*` -> Database Service (MongoDB backend)
- `/verify*` -> Verification Service