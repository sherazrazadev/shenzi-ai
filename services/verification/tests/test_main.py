import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_verify_email():
    response = client.post("/verify-email", json={"email": "test@example.com"})
    assert response.status_code == 200

def test_verify_invalid_email():
    response = client.post("/verify-email", json={"email": "invalid-email"})
    assert response.status_code == 400