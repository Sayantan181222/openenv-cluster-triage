# ── OpenEnv: Distributed Data Cluster Triage ──────────────────────────────
# Serves:
#   • OpenEnv HTTP API:  POST /reset, POST /step, GET /state, GET /health
#   • HTML Frontend:     GET  /  (interactive dashboard)
# Port: 7860 (required for Hugging Face Spaces)
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.10.14-slim-bookworm

WORKDIR /app

# Apply latest security patches
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy environment source
COPY . .

# Health check — confirms the FastAPI server is live
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run the FastAPI server (app:app = the `app` FastAPI instance in app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

