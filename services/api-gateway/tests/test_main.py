import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_gateway_unknown():
    response = client.get("/unknown")
    assert response.status_code == 404