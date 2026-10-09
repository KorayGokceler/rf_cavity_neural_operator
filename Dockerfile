# RF Cavity Neural Solver — web service (API + built frontend), CPU inference by default.
#   docker compose up --build        (model checkpoint in ./models, see docker-compose.yml)

# ── frontend ──────────────────────────────────────────────────────────────
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ── backend ───────────────────────────────────────────────────────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch \
    && pip install --no-cache-dir -r requirements.txt -r requirements-web.txt
COPY src/ src/
COPY --from=web /web/dist web/dist
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
ENV RFCAV_DEVICE=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/info')"
CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
