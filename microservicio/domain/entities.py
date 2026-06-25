from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConsultaClinica:
    """Variables clínicas estructuradas de un paciente. Sin dependencias externas."""
    edad:                    Optional[int]   = None
    sexo:                    Optional[str]   = None
    peso_kg:                 Optional[float] = None
    talla_cm:                Optional[float] = None
    presion_sistolica:       Optional[int]   = None
    presion_diastolica:      Optional[int]   = None
    glucosa_mg_dl:           Optional[int]   = None
    temperatura_c:           Optional[float] = None
    frecuencia_cardiaca_bpm: Optional[int]   = None
    duracion_sintomas_dias:  Optional[int]   = None
    categoria_sintoma:       Optional[str]   = None


@dataclass
class ResultadoExtraccion:
    campos:              ConsultaClinica
    campos_no_extraidos: list


@dataclass
class ResultadoAnomalia:
    es_anomalia:  bool
    score:        float
    nivel_riesgo: str   # "normal" | "sospechoso" | "anomalo"


@dataclass
class Inferencia:
    tipo:        str            # "extraccion" | "anomalia" | "completa"
    input_json:  dict
    output_json: dict
    id:          str            = field(default_factory=lambda: "")
    score:       Optional[float]= None
    es_anomalia: Optional[bool] = None
    created_at:  datetime       = field(default_factory=datetime.utcnow)
