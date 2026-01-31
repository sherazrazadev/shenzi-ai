FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app
ENV PYTHONPATH=/app

# Allow choosing which app module to run via APP_MODULE environment variable (defaults to app.main)
# To run the Ollama-backed service set: APP_MODULE=app.main_ollama
CMD uvicorn ${APP_MODULE:-app.main}:app --host 0.0.0.0 --port 8000