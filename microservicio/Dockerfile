# ── Imagen base ───────────────────────────────────────────────────────────────
# slim = sin compilador C; los wheels de sklearn/numpy se instalan como binarios
FROM python:3.11-slim

# ── Variables de entorno del sistema ──────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# ── Dependencias del sistema para psycopg2 ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ───────────────────────────────────────────────────────
# Copiamos requirements primero para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código de la aplicación ───────────────────────────────────────────────────
# Nota: el modelo NER se monta desde ../models/ner_clinico_es/ en el volumen.
# No se requiere descarga de modelos preentrenados.
COPY domain/        ./domain/
COPY application/   ./application/
COPY infrastructure/ ./infrastructure/

# ── Puerto y comando ──────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "infrastructure.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
