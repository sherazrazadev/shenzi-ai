from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os

app = FastAPI(title="API Gateway")

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth:8000")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent:8000")
VERIFICATION_SERVICE_URL = os.getenv("VERIFICATION_SERVICE_URL", "http://verification:8000")
DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL", "http://database:8000")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway(request: Request, path: str):
    # Routing based on path
    if path.startswith("auth"):
        url = f"{AUTH_SERVICE_URL}/{path}"
    elif path.startswith("call-ai"):
        url = f"{AGENT_SERVICE_URL}/{path.replace('call-ai', 'agent', 1)}"
    elif path.startswith("api"):
        url = f"{DATABASE_SERVICE_URL}/{path.replace('api', 'database', 1)}"
    elif path.startswith("verify"):
        url = f"{VERIFICATION_SERVICE_URL}/{path.replace('verify', 'verification', 1)}"
    else:
        raise HTTPException(status_code=404, detail="Service not found")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=await request.body()
            )
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))