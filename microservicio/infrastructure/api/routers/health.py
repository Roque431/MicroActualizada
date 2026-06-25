from fastapi import APIRouter
from infrastructure.api.schemas import HealthResponse

router = APIRouter(tags=["Sistema"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck del microservicio",
    description="Verifica que la API está viva. Usado por Docker para `depends_on: service_healthy`.",
)
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        modelos={
            "isolation_forest": "isolation_forest.joblib",
            "extractor_ner":    "extractor_ner_config.joblib",
            "scaler":           "scaler_if.joblib",
        },
    )
