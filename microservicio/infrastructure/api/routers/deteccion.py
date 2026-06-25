from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from application.dto import DeteccionRequestDTO
from application.use_cases import DetectarAnomaliaUseCase
from infrastructure.adapters.isolation_forest_adapter import IsolationForestAdapter
from infrastructure.adapters.postgres_repository import PostgresRepository
from infrastructure.api.dependencies import get_detector
from infrastructure.api.schemas import DeteccionRequest, DeteccionResponse
from infrastructure.db.session import get_db

router = APIRouter(tags=["Detección de Anomalías"])


@router.post(
    "/deteccion-anomalias",
    response_model=DeteccionResponse,
    summary="Detecta anomalías epidemiológicas en variables clínicas",
    description=(
        "Recibe variables clínicas estructuradas y corre el Isolation Forest entrenado. "
        "Devuelve `es_anomalia`, `score` (más negativo = más anómalo) y `nivel_riesgo` "
        "(`normal` / `sospechoso` / `anomalo`). Persiste la inferencia."
    ),
)
def detectar(
    body:     DeteccionRequest,
    db:       Session               = Depends(get_db),
    detector: IsolationForestAdapter = Depends(get_detector),
):
    repo    = PostgresRepository(db)
    use_case = DetectarAnomaliaUseCase(detector, repo)
    result   = use_case.ejecutar(DeteccionRequestDTO(
        edad=body.edad,
        peso_kg=body.peso_kg,
        talla_cm=body.talla_cm,
        presion_sistolica=body.presion_sistolica,
        presion_diastolica=body.presion_diastolica,
        glucosa_mg_dl=body.glucosa_mg_dl,
        temperatura_c=body.temperatura_c,
        frecuencia_cardiaca_bpm=body.frecuencia_cardiaca_bpm,
        duracion_sintomas_dias=body.duracion_sintomas_dias,
        categoria_sintoma=body.categoria_sintoma,
    ))

    return DeteccionResponse(
        inferencia_id=result.inferencia_id,
        es_anomalia=result.es_anomalia,
        score=result.score,
        nivel_riesgo=result.nivel_riesgo,
        created_at=result.created_at,
    )
