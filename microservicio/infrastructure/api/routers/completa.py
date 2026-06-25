from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from application.use_cases import ConsultaCompletaUseCase
from infrastructure.adapters.extractor_adapter import ExtractorAdapter
from infrastructure.adapters.isolation_forest_adapter import IsolationForestAdapter
from infrastructure.adapters.postgres_repository import PostgresRepository
from infrastructure.api.dependencies import get_detector, get_extractor
from infrastructure.api.schemas import (
    AnomaliaResumen,
    ConsultaCompletaRequest,
    ConsultaCompletaResponse,
)
from infrastructure.db.session import get_db

router = APIRouter(tags=["Consulta Completa"])


@router.post(
    "/consulta-completa",
    response_model=ConsultaCompletaResponse,
    summary="Extracción + detección de anomalías en una sola llamada",
    description=(
        "Endpoint principal para la app móvil. Recibe texto libre de dictado médico, "
        "ejecuta extracción NER y luego detección de anomalías con Isolation Forest. "
        "Si faltan campos para el modelo, devuelve solo la extracción con una advertencia."
    ),
)
def consulta_completa(
    body:      ConsultaCompletaRequest,
    db:        Session               = Depends(get_db),
    extractor: ExtractorAdapter      = Depends(get_extractor),
    detector:  IsolationForestAdapter = Depends(get_detector),
):
    repo     = PostgresRepository(db)
    use_case = ConsultaCompletaUseCase(extractor, detector, repo)
    result   = use_case.ejecutar(body.texto)

    anomalia_resp = None
    if result.anomalia:
        anomalia_resp = AnomaliaResumen(
            es_anomalia=result.anomalia["es_anomalia"],
            score=result.anomalia["score"],
            nivel_riesgo=result.anomalia["nivel_riesgo"],
        )

    return ConsultaCompletaResponse(
        inferencia_id=result.inferencia_id,
        extraccion=result.extraccion,
        anomalia=anomalia_resp,
        campos_no_extraidos=result.campos_no_extraidos,
        advertencia=result.advertencia,
        created_at=result.created_at,
    )
