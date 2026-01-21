import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Agent Service is running"}

def test_get_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert response.json() == {"tasks": []}

def test_create_task():
    task = {"title": "Test Task", "description": "A test task"}
    response = client.post("/tasks", json=task)
    assert response.status_code == 200
    assert "Task created" in response.json()["message"]