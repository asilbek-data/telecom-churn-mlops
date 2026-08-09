# Container for the scoring API.  Build: docker build -t churn-api .
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY models ./models

ENV MODEL_PATH=/app/models/model.pkl
ENV PORT=8000
EXPOSE 8000

# Render (and most hosts) inject $PORT at runtime
CMD uvicorn main:app --app-dir /app/api --host 0.0.0.0 --port ${PORT}
