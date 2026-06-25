"""
Adapter que implementa ExtractorPort usando Claude API (Anthropic).
El LLM entiende el texto libre sin reglas fijas — el médico puede decir
"peso de 78 kg", "pesa 78 kilos", "78 kg de masa corporal", o "carga
aproximadamente 78" y el modelo extrae el valor correctamente en todos los casos.

Requiere: ANTHROPIC_API_KEY en variables de entorno.
"""
import json
import os

import anthropic

from domain.entities import ConsultaClinica, ResultadoExtraccion
from domain.ports import ExtractorPort

_SYSTEM_PROMPT = """Eres un extractor especializado en variables clínicas de consultas médicas en español.
Tu única tarea es leer el texto de una transcripción médica y devolver un JSON con las variables clínicas.

Reglas absolutas:
- Devuelve SOLO el JSON, sin texto adicional, sin explicaciones, sin markdown.
- Si una variable no aparece en el texto, pon null.
- Para talla en metros (ej. "1.72 m"), conviértela a centímetros (172.0).
- Para sexo, devuelve "M" o "F". Interpreta: señor/hombre/masculino/niño/paciente masculino → M.
  Señora/mujer/femenino/niña/paciente femenina → F.
- Para categoria_sintoma elige la más apropiada entre:
  Respiratorio, Gastrointestinal, Hipertensión, Diabetes, Vacunación,
  Nutrición, Embarazo, Traumatología, Dermatológico, Infeccioso/Vectorial.
"""

_USER_TEMPLATE = """Extrae las variables clínicas de este texto médico:

"{texto}"

Devuelve este JSON exacto (reemplaza null si encuentras el valor):
{{
  "edad": null,
  "sexo": null,
  "peso_kg": null,
  "talla_cm": null,
  "presion_sistolica": null,
  "presion_diastolica": null,
  "glucosa_mg_dl": null,
  "temperatura_c": null,
  "frecuencia_cardiaca_bpm": null,
  "duracion_sintomas_dias": null,
  "categoria_sintoma": null
}}"""

_TODOS_LOS_CAMPOS = [
    "edad", "sexo", "peso_kg", "talla_cm",
    "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c",
    "frecuencia_cardiaca_bpm", "duracion_sintomas_dias",
    "categoria_sintoma",
]


class LLMExtractorAdapter(ExtractorPort):
    """
    Extractor basado en Claude API. No usa reglas — el LLM entiende
    cualquier variante de fraseo médico en español.
    Instanciado una sola vez en el arranque (singleton).
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada")
        self._client = anthropic.Anthropic(api_key=api_key)

    def extraer(self, texto: str) -> ResultadoExtraccion:
        response = self._client.messages.create(
            model="claude-haiku-4-5",   # rápido y económico para extracción
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _USER_TEMPLATE.format(texto=texto),
            }],
        )

        raw = response.content[0].text.strip()

        # Eliminar posibles bloques markdown si el modelo los incluyó
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            datos = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: devolver todo vacío si el JSON está malformado
            datos = {c: None for c in _TODOS_LOS_CAMPOS}

        # Construir entidad de dominio
        campos = ConsultaClinica(
            edad=self._to_int(datos.get("edad")),
            sexo=datos.get("sexo"),
            peso_kg=self._to_float(datos.get("peso_kg")),
            talla_cm=self._to_float(datos.get("talla_cm")),
            presion_sistolica=self._to_int(datos.get("presion_sistolica")),
            presion_diastolica=self._to_int(datos.get("presion_diastolica")),
            glucosa_mg_dl=self._to_int(datos.get("glucosa_mg_dl")),
            temperatura_c=self._to_float(datos.get("temperatura_c")),
            frecuencia_cardiaca_bpm=self._to_int(datos.get("frecuencia_cardiaca_bpm")),
            duracion_sintomas_dias=self._to_int(datos.get("duracion_sintomas_dias")),
            categoria_sintoma=datos.get("categoria_sintoma"),
        )

        no_extraidos = [
            f for f in _TODOS_LOS_CAMPOS
            if getattr(campos, f) is None
        ]

        return ResultadoExtraccion(campos=campos, campos_no_extraidos=no_extraidos)

    @staticmethod
    def _to_int(v):
        try:
            return int(float(v)) if v is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_float(v):
        try:
            return round(float(v), 1) if v is not None else None
        except (ValueError, TypeError):
            return None
