"""
Adapter que implementa AnomalyDetectorPort cargando el Isolation Forest
y el scaler entrenados desde models/.
"""
import os
import unicodedata

import joblib
import numpy as np

from domain.entities import ConsultaClinica, ResultadoAnomalia
from domain.ports import AnomalyDetectorPort

# Umbrales para nivel_riesgo (basados en distribución de scores del entrenamiento)
_UMBRAL_SOSPECHOSO = -0.05
_UMBRAL_ANOMALO    = -0.15

_FEATURE_ORDER = [
    "edad", "peso_kg", "talla_cm",
    "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c",
    "frecuencia_cardiaca_bpm", "duracion_sintomas_dias",
    "categoria_sintoma_enc",
]


def _norm(s: str) -> str:
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode().lower().strip()


class IsolationForestAdapter(AnomalyDetectorPort):
    """
    Carga modelo + scaler una sola vez en el arranque.
    Thread-safe para predicción (sklearn predict es stateless tras fit).
    """

    def __init__(self, model_dir: str):
        modelo_path = os.path.join(model_dir, "isolation_forest.joblib")
        scaler_path = os.path.join(model_dir, "scaler_if.joblib")
        meta_path   = os.path.join(model_dir, "isolation_forest_meta.joblib")

        self._modelo = joblib.load(modelo_path)
        self._scaler = joblib.load(scaler_path)
        meta         = joblib.load(meta_path)

        # Reconstruir mapeo de categoría → índice
        self._clases = [_norm(c) for c in meta["categoria_sintoma_classes"]]

    def _encode_categoria(self, categoria: str) -> int:
        normalizada = _norm(categoria)
        try:
            return self._clases.index(normalizada)
        except ValueError:
            # Categoría desconocida → usar 0 (Dermatológico, el menos frecuente)
            return 0

    def detectar(self, consulta: ConsultaClinica) -> ResultadoAnomalia:
        enc = self._encode_categoria(consulta.categoria_sintoma or "")
        vector = np.array([[
            consulta.edad,
            consulta.peso_kg,
            consulta.talla_cm,
            consulta.presion_sistolica,
            consulta.presion_diastolica,
            consulta.glucosa_mg_dl,
            consulta.temperatura_c,
            consulta.frecuencia_cardiaca_bpm,
            consulta.duracion_sintomas_dias,
            enc,
        ]], dtype=float)

        vector_scaled = self._scaler.transform(vector)
        pred          = self._modelo.predict(vector_scaled)[0]   # -1 o 1
        score         = float(self._modelo.decision_function(vector_scaled)[0])
        es_anomalia   = bool(pred == -1)   # np.bool_ → Python bool para serialización JSON

        if not es_anomalia:
            nivel = "normal"
        elif score > _UMBRAL_SOSPECHOSO:
            nivel = "sospechoso"
        else:
            nivel = "anomalo"

        return ResultadoAnomalia(es_anomalia=es_anomalia, score=round(score, 4), nivel_riesgo=nivel)
