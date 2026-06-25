from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from application.use_cases import ConsultarHistorialUseCase
from infrastructure.adapters.postgres_repository import PostgresRepository
from infrastructure.api.schemas import InferenciasResponse, InferenciaResumen
from infrastructure.db.session import get_db

router = APIRouter(tags=["Historial"])


@router.get(
    "/inferencias",
    response_model=InferenciasResponse,
    summary="Historial de inferencias",
    description=(
        "Devuelve el historial de todas las inferencias persistidas. "
        "Filtra por tipo (`extraccion`, `anomalia`, `completa`) y rango de fechas."
    ),
)
def listar_inferencias(
    tipo:   Optional[str]      = Query(None, description="extraccion | anomalia | completa"),
    desde:  Optional[datetime] = Query(None, description="Fecha/hora de inicio ISO 8601"),
    hasta:  Optional[datetime] = Query(None, description="Fecha/hora de fin ISO 8601"),
    limit:  int                = Query(50,   ge=1, le=200),
    offset: int                = Query(0,    ge=0),
    db:     Session            = Depends(get_db),
):
    repo     = PostgresRepository(db)
    use_case = ConsultarHistorialUseCase(repo)
    result   = use_case.ejecutar(tipo, desde, hasta, limit, offset)

    return InferenciasResponse(
        total=result.total,
        inferencias=[InferenciaResumen(**i) for i in result.inferencias],
    )
