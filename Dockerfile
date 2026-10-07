FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY rag ./rag
COPY sql ./sql
COPY data ./data
COPY eval ./eval
# Cloud Run sets $PORT. One worker per instance; scale with Cloud Run concurrency instead.
CMD exec uvicorn rag.api:app --host 0.0.0.0 --port ${PORT:-8080}
