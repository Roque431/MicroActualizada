"""
Esquemas Pydantic para validación de entrada/salida.
SOLO en infrastructure/ — nunca en domain/ ni application/.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Requests ───────────────────────────────────────────────────────────────────

class ExtraccionRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        description="Transcripción de voz del médico en español.",
        examples=["Paciente masculino de 45 años, presión 125/82 mmHg, glucosa 98 mg/dl."],
    )


class DeteccionRequest(BaseModel):
    edad:                    int   = Field(..., ge=0,   le=120, description="Edad en años")
    peso_kg:                 float = Field(..., gt=0,          description="Peso en kilogramos")
    talla_cm:                float = Field(..., gt=0,          description="Talla en centímetros")
    presion_sistolica:       int   = Field(..., ge=50, le=300, description="Presión sistólica mmHg")
    presion_diastolica:      int   = Field(..., ge=30, le=200, description="Presión diastólica mmHg")
    glucosa_mg_dl:           int   = Field(..., ge=20, le=800, description="Glucosa mg/dL")
    temperatura_c:           float = Field(..., ge=30, le=45,  description="Temperatura °C")
    frecuencia_cardiaca_bpm: int   = Field(..., ge=20, le=300, description="Frecuencia cardíaca lpm")
    duracion_sintomas_dias:  int   = Field(..., ge=0,          description="Duración de síntomas en días")
    categoria_sintoma:       str   = Field(..., description="Categoría clínica del síntoma principal")


class ConsultaCompletaRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        description="Transcripción completa del dictado médico en español.",
    )


# ── Responses ──────────────────────────────────────────────────────────────────

class ExtraccionResponse(BaseModel):
    inferencia_id:       str
    campos_extraidos:    Dict[str, Any]
    campos_no_extraidos: List[str]
    created_at:          datetime


class DeteccionResponse(BaseModel):
    inferencia_id: str
    es_anomalia:   bool
    score:         float = Field(description="Anomaly score: más negativo = más anómalo")
    nivel_riesgo:  str   = Field(description="'normal' | 'sospechoso' | 'anomalo'")
    created_at:    datetime


class AnomaliaResumen(BaseModel):
    es_anomalia:  bool
    score:        float
    nivel_riesgo: str


class ConsultaCompletaResponse(BaseModel):
    inferencia_id:       str
    extraccion:          Dict[str, Any]
    anomalia:            Optional[AnomaliaResumen]
    campos_no_extraidos: List[str]
    advertencia:         Optional[str] = None
    created_at:          datetime


class InferenciaResumen(BaseModel):
    id:          str
    tipo:        str
    score:       Optional[float]
    es_anomalia: Optional[bool]
    created_at:  str
    output_json: Dict[str, Any]


class InferenciasResponse(BaseModel):
    total:       int
    inferencias: List[InferenciaResumen]


class HealthResponse(BaseModel):
    status:  str
    version: str
    modelos: Dict[str, str]
