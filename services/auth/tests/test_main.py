import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_login():
    response = client.post("/token", json={"username": "admin", "password": "password"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid():
    response = client.post("/token", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 400

def test_verify_token():
    # First login
    response = client.post("/token", json={"username": "admin", "password": "password"})
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/verify", headers=headers)
    assert response.status_code == 200