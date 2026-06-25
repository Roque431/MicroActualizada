"""Data Transfer Objects entre capas. No tienen lógica — solo transportan datos."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ExtraccionRequestDTO:
    texto: str


@dataclass
class DeteccionRequestDTO:
    edad:                    int
    peso_kg:                 float
    talla_cm:                float
    presion_sistolica:       int
    presion_diastolica:      int
    glucosa_mg_dl:           int
    temperatura_c:           float
    frecuencia_cardiaca_bpm: int
    duracion_sintomas_dias:  int
    categoria_sintoma:       str


@dataclass
class ExtraccionResponseDTO:
    inferencia_id:       str
    campos_extraidos:    dict
    campos_no_extraidos: list
    created_at:          datetime


@dataclass
class DeteccionResponseDTO:
    inferencia_id: str
    es_anomalia:   bool
    score:         float
    nivel_riesgo:  str
    created_at:    datetime


@dataclass
class ConsultaCompletaResponseDTO:
    inferencia_id:       str
    extraccion:          dict
    anomalia:            Optional[dict]
    campos_no_extraidos: list
    advertencia:         Optional[str]
    created_at:          datetime


@dataclass
class InferenciaListaDTO:
    total:       int
    inferencias: list
