# EpiDiagnostix-Mayab — Microservicio ML

Microservicio de Machine Learning para detección de anomalías epidemiológicas
y extracción de variables clínicas desde texto libre en español.

## Arquitectura Hexagonal (Ports & Adapters)

```
microservicio/
│
├── domain/                  ← Núcleo. SIN dependencias externas
│   ├── entities.py          │  ConsultaClinica, ResultadoExtraccion,
│   └── ports.py             │  ResultadoAnomalia, Inferencia
│                            │  ExtractorPort, AnomalyDetectorPort,
│                            │  RepositorioInferenciasPort (ABCs)
│
├── application/             ← Casos de uso. Solo depende de domain/
│   ├── dto.py               │  Objetos de transferencia entre capas
│   └── use_cases.py         │  ExtraerVariablesUseCase
│                            │  DetectarAnomaliaUseCase
│                            │  ConsultaCompletaUseCase
│                            │  ConsultarHistorialUseCase
│
└── infrastructure/          ← Adapters. Depende de application/ y librerías externas
    ├── api/
    │   ├── main.py          │  FastAPI app + lifespan (startup/shutdown)
    │   ├── schemas.py       │  Pydantic — validación de entrada/salida
    │   ├── dependencies.py  │  Singletons de adapters ML
    │   └── routers/
    │       ├── health.py    │  GET  /health
    │       ├── extraccion.py│  POST /extraccion
    │       ├── deteccion.py │  POST /deteccion-anomalias
    │       ├── completa.py  │  POST /consulta-completa
    │       └── inferencias.py│ GET  /inferencias
    ├── adapters/
    │   ├── extractor_adapter.py         ← Implementa ExtractorPort (regex/spaCy)
    │   ├── isolation_forest_adapter.py  ← Implementa AnomalyDetectorPort (IF + scaler)
    │   └── postgres_repository.py      ← Implementa RepositorioInferenciasPort
    └── db/
        ├── models.py        │  ORM SQLAlchemy — tabla inferencias
        └── session.py       │  engine + SessionLocal + get_db()
```

## Requisitos previos

- Docker Desktop instalado y corriendo
- Los modelos entrenados en `../models/` (generados por los notebooks de entrenamiento):
  - `isolation_forest.joblib`
  - `scaler_if.joblib`
  - `isolation_forest_meta.joblib`

## Levantar el stack

```bash
# 1. Ir al directorio del microservicio
cd microservicio

# 2. Crear el archivo .env con la contraseña de PostgreSQL
cp .env.example .env
# Editar .env y cambiar POSTGRES_PASSWORD por una contraseña segura

# 3. Construir y levantar
docker-compose up --build

# Con detach (fondo):
docker-compose up --build -d
```

La API queda disponible en `http://localhost:8000`
Documentación Swagger: `http://localhost:8000/docs`
Documentación ReDoc:   `http://localhost:8000/redoc`

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `POSTGRES_PASSWORD` | **Sí** | — | Contraseña de PostgreSQL |
| `POSTGRES_USER` | No | `epiadmin` | Usuario de PostgreSQL |
| `POSTGRES_DB` | No | `epidiagnostix` | Nombre de la base de datos |
| `MODEL_DIR` | No | `/app/models` | Ruta a los modelos dentro del contenedor |

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Healthcheck del contenedor |
| `POST` | `/extraccion` | Texto libre → variables clínicas estructuradas |
| `POST` | `/deteccion-anomalias` | Variables estructuradas → predicción IF |
| `POST` | `/consulta-completa` | Texto libre → extracción + detección en una llamada |
| `GET` | `/inferencias` | Historial con filtros opcionales |

## Correr las pruebas de integración

```bash
# Con el stack corriendo:
cd microservicio
pip install httpx pytest
pytest tests/test_integration.py -v
```

## Importar colección Postman

Importar el archivo `postman/epidiagnostix_ml_collection.json` en Postman.
La variable `base_url` apunta a `http://localhost:8000` por defecto.

## Detener el stack

```bash
docker-compose down          # detener sin borrar datos
docker-compose down -v       # detener y borrar volumen de PostgreSQL
```
