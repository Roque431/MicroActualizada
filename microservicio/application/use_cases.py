from dataclasses import asdict
from datetime import datetime
from typing import Optional

from domain.entities import ConsultaClinica, Inferencia
from domain.ports import (
    AnomalyDetectorPort,
    ExtractorPort,
    RepositorioInferenciasPort,
)
from application.dto import (
    ConsultaCompletaResponseDTO,
    DeteccionRequestDTO,
    DeteccionResponseDTO,
    ExtraccionRequestDTO,
    ExtraccionResponseDTO,
    InferenciaListaDTO,
)

# Campos que el Isolation Forest necesita obligatoriamente
_CAMPOS_REQUERIDOS_IF = [
    "edad", "peso_kg", "talla_cm", "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c", "frecuencia_cardiaca_bpm",
    "duracion_sintomas_dias", "categoria_sintoma",
]


class ExtraerVariablesUseCase:
    def __init__(self, extractor: ExtractorPort, repo: RepositorioInferenciasPort):
        self._extractor = extractor
        self._repo = repo

    def ejecutar(self, req: ExtraccionRequestDTO) -> ExtraccionResponseDTO:
        resultado = self._extractor.extraer(req.texto)
        campos_dict = asdict(resultado.campos)

        inferencia = self._repo.guardar(Inferencia(
            tipo="extraccion",
            input_json={"texto": req.texto},
            output_json={"campos": campos_dict,
                         "campos_no_extraidos": resultado.campos_no_extraidos},
        ))

        return ExtraccionResponseDTO(
            inferencia_id=inferencia.id,
            campos_extraidos=campos_dict,
            campos_no_extraidos=resultado.campos_no_extraidos,
            created_at=inferencia.created_at,
        )


class DetectarAnomaliaUseCase:
    def __init__(self, detector: AnomalyDetectorPort, repo: RepositorioInferenciasPort):
        self._detector = detector
        self._repo = repo

    def ejecutar(self, req: DeteccionRequestDTO) -> DeteccionResponseDTO:
        consulta = ConsultaClinica(
            edad=req.edad,
            peso_kg=req.peso_kg,
            talla_cm=req.talla_cm,
            presion_sistolica=req.presion_sistolica,
            presion_diastolica=req.presion_diastolica,
            glucosa_mg_dl=req.glucosa_mg_dl,
            temperatura_c=req.temperatura_c,
            frecuencia_cardiaca_bpm=req.frecuencia_cardiaca_bpm,
            duracion_sintomas_dias=req.duracion_sintomas_dias,
            categoria_sintoma=req.categoria_sintoma,
        )
        resultado = self._detector.detectar(consulta)

        inferencia = self._repo.guardar(Inferencia(
            tipo="anomalia",
            input_json=asdict(consulta),
            output_json={
                "es_anomalia": resultado.es_anomalia,
                "score": resultado.score,
                "nivel_riesgo": resultado.nivel_riesgo,
            },
            score=resultado.score,
            es_anomalia=resultado.es_anomalia,
        ))

        return DeteccionResponseDTO(
            inferencia_id=inferencia.id,
            es_anomalia=resultado.es_anomalia,
            score=resultado.score,
            nivel_riesgo=resultado.nivel_riesgo,
            created_at=inferencia.created_at,
        )


class ConsultaCompletaUseCase:
    def __init__(
        self,
        extractor: ExtractorPort,
        detector:  AnomalyDetectorPort,
        repo:      RepositorioInferenciasPort,
    ):
        self._extractor = extractor
        self._detector  = detector
        self._repo      = repo

    def ejecutar(self, texto: str) -> ConsultaCompletaResponseDTO:
        # 1. Extracción
        resultado_ext = self._extractor.extraer(texto)
        campos = asdict(resultado_ext.campos)

        # 2. Verificar si se pueden alimentar al IF
        faltantes = [
            c for c in _CAMPOS_REQUERIDOS_IF if campos.get(c) is None
        ]

        anomalia_dict = None
        advertencia   = None
        score         = None
        es_anomalia   = None

        if not faltantes:
            consulta = ConsultaClinica(**{k: campos[k] for k in _CAMPOS_REQUERIDOS_IF
                                         if k != "categoria_sintoma"},
                                      categoria_sintoma=campos["categoria_sintoma"])
            resultado_an = self._detector.detectar(consulta)
            anomalia_dict = {
                "es_anomalia": resultado_an.es_anomalia,
                "score": resultado_an.score,
                "nivel_riesgo": resultado_an.nivel_riesgo,
            }
            score       = resultado_an.score
            es_anomalia = resultado_an.es_anomalia
        else:
            advertencia = f"Detección omitida: faltan campos {faltantes}"

        inferencia = self._repo.guardar(Inferencia(
            tipo="completa",
            input_json={"texto": texto},
            output_json={
                "extraccion": campos,
                "anomalia": anomalia_dict,
                "campos_no_extraidos": resultado_ext.campos_no_extraidos,
            },
            score=score,
            es_anomalia=es_anomalia,
        ))

        return ConsultaCompletaResponseDTO(
            inferencia_id=inferencia.id,
            extraccion=campos,
            anomalia=anomalia_dict,
            campos_no_extraidos=resultado_ext.campos_no_extraidos,
            advertencia=advertencia,
            created_at=inferencia.created_at,
        )


class ConsultarHistorialUseCase:
    def __init__(self, repo: RepositorioInferenciasPort):
        self._repo = repo

    def ejecutar(
        self,
        tipo:   Optional[str],
        desde:  Optional[datetime],
        hasta:  Optional[datetime],
        limit:  int,
        offset: int,
    ) -> InferenciaListaDTO:
        inferencias, total = self._repo.listar(tipo, desde, hasta, limit, offset)
        return InferenciaListaDTO(total=total, inferencias=[
            {
                "id": i.id,
                "tipo": i.tipo,
                "score": i.score,
                "es_anomalia": i.es_anomalia,
                "created_at": i.created_at.isoformat(),
                "output_json": i.output_json,
            }
            for i in inferencias
        ])
