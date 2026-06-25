"""
Singletons de los adapters de ML — se inicializan en startup de FastAPI.

Jerarquía de extractores (de mayor a menor prioridad):
  1. NERExtractorAdapter    → modelo propio entrenado con spaCy (offline, sin costo)
  2. ExtractorAdapter       → regex/reglas (offline, fallback si NER no está listo)
  3. LLMExtractorAdapter    → Claude API (solo si ANTHROPIC_API_KEY está seteada)

Para la tesis: el objetivo es el nivel 1 (NER supervisado propio).
"""
import os
import logging
from pathlib import Path

from infrastructure.adapters.isolation_forest_adapter import IsolationForestAdapter

logger = logging.getLogger(__name__)

_extractor = None
_detector:  IsolationForestAdapter | None = None


def init_adapters(model_dir: str):
    global _extractor, _detector

    ner_path = Path(model_dir) / "ner_clinico_es"

    if ner_path.exists():
        # Prioridad 1: modelo NER propio (objetivo de tesis)
        from infrastructure.adapters.ner_extractor_adapter import NERExtractorAdapter
        _extractor = NERExtractorAdapter(model_dir)
        logger.info("Extractor: NER local (spaCy) — modelo propio entrenado")

    elif os.environ.get("ANTHROPIC_API_KEY"):
        # Prioridad 2: Claude API (si existe API key, sin NER)
        from infrastructure.adapters.llm_extractor_adapter import LLMExtractorAdapter
        _extractor = LLMExtractorAdapter()
        logger.info("Extractor: LLM (Claude API)")

    else:
        # Prioridad 3: regex — fallback offline siempre disponible
        from infrastructure.adapters.extractor_adapter import ExtractorAdapter
        _extractor = ExtractorAdapter()
        logger.info("Extractor: regex/spaCy — modo offline fallback")

    _detector = IsolationForestAdapter(model_dir)
    logger.info("Isolation Forest cargado desde %s", model_dir)


def get_extractor():
    return _extractor


def get_detector() -> IsolationForestAdapter:
    return _detector
