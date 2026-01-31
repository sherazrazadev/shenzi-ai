from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os

app = FastAPI(title="API Gateway")

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth:8000")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent:8000")
VERIFICATION_SERVICE_URL = os.getenv("VERIFICATION_SERVICE_URL", "http://verification:8000")
DATABASE_SERVICE_URL = os.getenv("DATABASE_SERVICE_URL", "http://database:8000")

# Use a long-lived AsyncClient so streaming generators can use it after the request handler returns
client = httpx.AsyncClient(timeout=60.0)

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await client.aclose()
    except Exception:
        pass


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway(request: Request, path: str):
    # Routing based on path
    if path.startswith("auth"):
        url = f"{AUTH_SERVICE_URL}/{path}"
    elif path.startswith("call-ai"):
        url = f"{AGENT_SERVICE_URL}/{path.replace('call-ai/', '', 1)}"
    elif path.startswith("chat"):
        url = f"{AGENT_SERVICE_URL}/{path}"
    elif path.startswith("api"):
        url = f"{DATABASE_SERVICE_URL}/{path.replace('api', 'database', 1)}"
    elif path.startswith("verify"):
        url = f"{VERIFICATION_SERVICE_URL}/{path.replace('verify', 'verification', 1)}"
    else:
        raise HTTPException(status_code=404, detail="Service not found")

    try:
        if path.startswith("chat/stream"):
            # Robust streaming proxy for /chat/stream
            from fastapi.responses import StreamingResponse
            req_body = await request.body()

            async def stream_gen():
                try:
                    async with client.stream(
                        method=request.method,
                        url=url,
                        headers=dict(request.headers),
                        content=req_body
                    ) as stream_response:
                        # Forward status and headers as needed (we default to SSE)
                        async for chunk in stream_response.aiter_bytes():
                            yield chunk
                except httpx.RequestError as re:
                    # Upstream request failed
                    print(f"API Gateway stream request error: {re}")
                    return
                except Exception as ex:
                    print(f"API Gateway stream error: {ex}")
                    return

            return StreamingResponse(stream_gen(), status_code=200, media_type="text/event-stream")
        else:
            # Normal request for all other routes
            response = await client.request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=await request.body()
            )
            try:
                json_body = response.json()
            except Exception:
                json_body = response.text
            return JSONResponse(status_code=response.status_code, content=json_body)
    except httpx.RequestError as e:
        print("API Gateway RequestError:", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        import traceback
        print("API Gateway Exception:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))