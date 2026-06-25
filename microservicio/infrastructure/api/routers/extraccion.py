from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from application.dto import ExtraccionRequestDTO
from application.use_cases import ExtraerVariablesUseCase
from infrastructure.adapters.extractor_adapter import ExtractorAdapter
from infrastructure.adapters.postgres_repository import PostgresRepository
from infrastructure.api.dependencies import get_extractor
from infrastructure.api.schemas import ExtraccionRequest, ExtraccionResponse
from infrastructure.db.session import get_db

router = APIRouter(tags=["Extracción NER"])


@router.post(
    "/extraccion",
    response_model=ExtraccionResponse,
    summary="Extrae variables clínicas desde texto libre",
    description=(
        "Recibe una transcripción de dictado médico en español y extrae las variables "
        "clínicas estructuradas usando el motor de reglas/regex. "
        "Persiste la inferencia en PostgreSQL."
    ),
)
def extraer(
    body:      ExtraccionRequest,
    db:        Session        = Depends(get_db),
    extractor: ExtractorAdapter = Depends(get_extractor),
):
    repo    = PostgresRepository(db)
    use_case = ExtraerVariablesUseCase(extractor, repo)
    result   = use_case.ejecutar(ExtraccionRequestDTO(texto=body.texto))

    return ExtraccionResponse(
        inferencia_id=result.inferencia_id,
        campos_extraidos=result.campos_extraidos,
        campos_no_extraidos=result.campos_no_extraidos,
        created_at=result.created_at,
    )
