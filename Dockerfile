FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt
COPY backend/requirements-backend.txt /app/requirements-backend.txt

# Install Python deps (project + backend)
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements-backend.txt

# Copy code
COPY . /app

# Download WordNet data at build time to avoid runtime downloads
RUN python - <<'PY'\nimport nltk\nnltk.download('wordnet', quiet=True)\nnltk.download('omw-1.4', quiet=True)\nprint('WordNet data downloaded')\nPY

EXPOSE 8000

ENV PYTHONPATH=/app

CMD [\"uvicorn\", \"backend.app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n

