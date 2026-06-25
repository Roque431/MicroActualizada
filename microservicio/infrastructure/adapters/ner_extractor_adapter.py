"""
Adapter que implementa ExtractorPort usando el modelo NER local entrenado con spaCy.

- 100% offline — no requiere internet
- Gratuito — sin APIs de pago
- Ligero — ~10 MB en disco, <100 MB en RAM
- Generaliza por contexto, no por reglas rígidas

El modelo se entrena con: python entrenar_ner.py
Se guarda en: models/ner_clinico_es/
"""
import os
import unicodedata
from pathlib import Path

import spacy

from domain.entities import ConsultaClinica, ResultadoExtraccion
from domain.ports import ExtractorPort

_TODOS_LOS_CAMPOS = [
    "edad", "sexo", "peso_kg", "talla_cm",
    "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c",
    "frecuencia_cardiaca_bpm", "duracion_sintomas_dias",
    "categoria_sintoma",
]

_SEXO_MAP = {
    "masculino": "M", "hombre": "M", "varon": "M", "nino": "M",
    "senor": "M", "paciente masculino": "M",
    "femenino": "F", "mujer": "F", "femenina": "F", "nina": "F",
    "senora": "F", "paciente femenina": "F",
}

_CAT_MAP = {
    "gastrointestinal": "Gastrointestinal",
    "respiratorio":     "Respiratorio",
    "hipertension":     "Hipertensión",
    "diabetes":         "Diabetes",
    "vacunacion":       "Vacunación",
    "nutricion":        "Nutrición",
    "embarazo":         "Embarazo",
    "traumatologia":    "Traumatología",
    "dermatologico":    "Dermatológico",
    "infeccioso":       "Infeccioso/Vectorial",
    "vectorial":        "Infeccioso/Vectorial",
    "dengue":           "Infeccioso/Vectorial",
    "malaria":          "Infeccioso/Vectorial",
    "paludismo":        "Infeccioso/Vectorial",
    "chikungunya":      "Infeccioso/Vectorial",
    "zika":             "Infeccioso/Vectorial",
}

# Fallback de keywords cuando NER no detecta CATEGORIA
_CAT_KEYWORDS = {
    "Gastrointestinal":     ["gastrointestinal","diarrea","vomito","nausea","dolor abdominal","gastro"],
    "Respiratorio":         ["respiratorio","tos","dificultad para respirar","congestion","gripe","bronquitis","neumonia"],
    "Hipertensión":         ["hipertension","hipertenso","hipertensa","presion alta","vision borrosa"],
    "Diabetes":             ["diabetes","diabetico","diabetica","glucemia alta","azucar alta"],
    "Vacunación":           ["vacunacion","vacuna","esquema de vacunacion","postvacuna"],
    "Nutrición":            ["nutricion","desnutricion","palidez","anemia","debilidad"],
    "Embarazo":             ["embarazo","embarazada","gestacion","prenatal"],
    "Traumatología":        ["traumatologia","esguince","fractura","herida","contusion"],
    "Dermatológico":        ["dermatologico","dermatitis","erupcion","sarpullido"],
    "Infeccioso/Vectorial": ["dengue","chikungunya","zika","malaria","paludismo","leptospirosis",
                             "fiebre dengue","sospechoso de dengue","infeccioso"],
}


def _inferir_categoria(texto: str) -> str | None:
    """Busca categoría por keywords si NER no la detectó."""
    t = _norm(texto)
    best, best_score = None, 0
    for cat, kws in _CAT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best, best_score = cat, score
    return best if best_score > 0 else None


def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def _to_int(v):
    try:
        return int(float(v.replace(",", "."))) if v else None
    except (ValueError, TypeError):
        return None


def _to_float(v):
    try:
        return round(float(v.replace(",", ".")), 1) if v else None
    except (ValueError, TypeError):
        return None


class NERExtractorAdapter(ExtractorPort):
    """
    Extractor clínico basado en el modelo NER entrenado localmente.
    Aprende de contexto — no usa reglas rígidas.
    """

    def __init__(self, model_dir: str):
        model_path = Path(model_dir) / "ner_clinico_es"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo NER no encontrado en {model_path}. "
                "Ejecuta primero: python entrenar_ner.py"
            )
        self._nlp = spacy.load(str(model_path))

    def extraer(self, texto: str) -> ResultadoExtraccion:
        doc = self._nlp(texto)
        resultado = {c: None for c in _TODOS_LOS_CAMPOS}

        # Acumular candidatos por etiqueta (tomar el primero encontrado)
        for ent in doc.ents:
            txt = ent.text.strip()
            lbl = ent.label_

            if lbl == "EDAD" and resultado["edad"] is None:
                v = _to_int(txt)
                if v and 0 < v <= 120:
                    resultado["edad"] = v

            elif lbl == "SEXO" and resultado["sexo"] is None:
                resultado["sexo"] = _SEXO_MAP.get(_norm(txt))

            elif lbl == "PESO_KG" and resultado["peso_kg"] is None:
                resultado["peso_kg"] = _to_float(txt)

            elif lbl == "TALLA_CM" and resultado["talla_cm"] is None:
                v = _to_float(txt)
                if v:
                    # Si el valor es <3, está en metros → convertir a cm
                    resultado["talla_cm"] = round(v * 100, 1) if v < 3.0 else v

            elif lbl == "PRESION_SIS" and resultado["presion_sistolica"] is None:
                # Puede venir como "120" o como "120/80" (si tokenizó junto)
                if "/" in txt:
                    partes = txt.split("/")
                    resultado["presion_sistolica"]  = _to_int(partes[0])
                    resultado["presion_diastolica"] = _to_int(partes[1])
                else:
                    resultado["presion_sistolica"] = _to_int(txt)

            elif lbl == "PRESION_DIA" and resultado["presion_diastolica"] is None:
                resultado["presion_diastolica"] = _to_int(txt)

            elif lbl == "GLUCOSA" and resultado["glucosa_mg_dl"] is None:
                resultado["glucosa_mg_dl"] = _to_int(txt)

            elif lbl == "TEMPERATURA" and resultado["temperatura_c"] is None:
                resultado["temperatura_c"] = _to_float(txt)

            elif lbl == "FREC_CARD":
                # El modelo a veces etiqueta "120/80" como FREC_CARD por confusión
                # Si parece presión arterial (X/Y con valores típicos), reclasificar
                if "/" in txt and resultado["presion_sistolica"] is None:
                    partes = txt.split("/")
                    try:
                        ps = int(partes[0])
                        pd = int(partes[1])
                        if 60 <= ps <= 250 and 40 <= pd <= 150:
                            resultado["presion_sistolica"]  = ps
                            resultado["presion_diastolica"] = pd
                            continue
                    except (ValueError, IndexError):
                        pass
                if resultado["frecuencia_cardiaca_bpm"] is None:
                    resultado["frecuencia_cardiaca_bpm"] = _to_int(txt)

            elif lbl == "DURACION" and resultado["duracion_sintomas_dias"] is None:
                resultado["duracion_sintomas_dias"] = _to_int(txt)

            elif lbl == "CATEGORIA" and resultado["categoria_sintoma"] is None:
                k = _norm(txt)
                resultado["categoria_sintoma"] = _CAT_MAP.get(k, txt)

        # Fallback: si NER no detectó categoría, buscar por keywords en el texto completo
        if resultado["categoria_sintoma"] is None:
            resultado["categoria_sintoma"] = _inferir_categoria(texto)

        campos = ConsultaClinica(**resultado)
        no_extraidos = [f for f in _TODOS_LOS_CAMPOS if getattr(campos, f) is None]
        return ResultadoExtraccion(campos=campos, campos_no_extraidos=no_extraidos)
