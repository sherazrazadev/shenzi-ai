from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI(title="Database Service", description="MongoDB backend service")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://mongodb:27017")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.shenzi_db

@app.on_event("startup")
async def startup_event():
    # Test connection
    await client.admin.command('ping')
    print("Connected to MongoDB")

@app.get("/")
async def root():
    return {"message": "Database Service is running"}

@app.get("/users")
async def get_users():
    users = await db.users.find().to_list(100)
    return {"users": users}

@app.post("/users")
async def create_user(user: dict):
    result = await db.users.insert_one(user)
    return {"inserted_id": str(result.inserted_id)}