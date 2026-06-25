import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.api.dependencies import init_adapters
from infrastructure.api.routers import completa, deteccion, extraccion, health, inferencias
from infrastructure.db.models import Base
from infrastructure.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    # 1. Crear tablas en PostgreSQL (equivale a Alembic init para el proyecto)
    Base.metadata.create_all(bind=engine)

    # 2. Cargar modelos ML desde el directorio configurado por variable de entorno
    model_dir = os.environ.get("MODEL_DIR", "/app/models")
    init_adapters(model_dir)

    yield
    # ── Shutdown (nada que limpiar) ───────────────────────────────────────────


app = FastAPI(
    title="EpiDiagnostix-Mayab — ML Microservicio",
    description=(
        "Microservicio de Machine Learning para detección de anomalías epidemiológicas "
        "y extracción de variables clínicas desde texto libre. "
        "Proyecto: EpiDiagnostix-Mayab — Comunidades de Chiapas."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health.router)
app.include_router(extraccion.router)
app.include_router(deteccion.router)
app.include_router(completa.router)
app.include_router(inferencias.router)
