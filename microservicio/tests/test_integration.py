"""
Pruebas de integración — asumen que el stack de docker-compose está corriendo.
Ejecutar con:
    cd microservicio
    docker-compose up --build -d
    pytest tests/test_integration.py -v
"""
import pytest
import httpx

BASE_URL = "http://localhost:8000"

# ── Los 5 estilos de test_extractor.py ────────────────────────────────────────
CASOS = [
    {
        "nombre": "Caso 1 — SOAP clásico",
        "texto": (
            "Consulta médica. Paciente masculino de 45 años de edad. "
            "Peso 82.0 kg, talla 170.5 cm. "
            "Motivo: dolor abdominal, diarrea, deshidratación, 5 días de evolución. "
            "Signos vitales: presión arterial de 125/82 mmHg, glucosa de 98 mg/dl, "
            "temperatura 37.2°C, frecuencia cardiaca de 91 latidos por minuto."
        ),
        "esperado": {
            "edad": 45, "sexo": "M", "presion_sistolica": 125,
            "presion_diastolica": 82, "glucosa_mg_dl": 98,
        },
    },
    {
        "nombre": "Caso 2 — Dictado rápido abreviado",
        "texto": (
            "Paciente femenina, 67 años. PA: 160/95, glucemia capilar 310 mg/dL. "
            "T: 36.8 grados. FC: 88 lpm. Peso 63.5 kg, estatura de 155.0 cm. "
            "Síntomas: fatiga y pérdida de peso desde hace 3 días."
        ),
        "esperado": {
            "edad": 67, "sexo": "F", "presion_sistolica": 160,
            "presion_diastolica": 95, "glucosa_mg_dl": 310,
        },
    },
    {
        "nombre": "Caso 3 — Coloquial comunidad",
        "texto": (
            "Se atiende a mujer de 32 años que lleva 7 días con tos y dificultad para respirar. "
            "Presión 118 sobre 74, azúcar en sangre 89, temperatura 38.5, pulso 102. "
            "Pesa 58 kilos, mide 162 cm."
        ),
        "esperado": {
            "edad": 32, "sexo": "F", "presion_sistolica": 118,
            "presion_diastolica": 74, "glucosa_mg_dl": 89,
        },
    },
    {
        "nombre": "Caso 4 — Pediátrico/vacunación",
        "texto": (
            "Paciente masculino, edad: 2 años. Consulta por fiebre leve postvacuna "
            "y control de esquema de vacunación con 1 día de evolución. "
            "Peso corporal: 12.5 kg, talla 82.0 centímetros. "
            "Tensión arterial 100/65 mmHg, temperatura de 37.8 grados, "
            "frecuencia cardiaca de 110 lpm, glucosa 88 mg/dl."
        ),
        "esperado": {
            "edad": 2, "sexo": "M", "presion_sistolica": 100,
            "presion_diastolica": 65, "glucosa_mg_dl": 88,
        },
    },
    {
        "nombre": "Caso 5 — Texto libre sin estructura",
        "texto": (
            "La señora tiene como 55 años, es hipertensa, refiere zumbido en los oídos "
            "y visión borrosa desde hace 4 días. Le tomé la presión y salió 170 sobre 100. "
            "Su azúcar estaba bien, 95. Temperatura normal 36.5. Pulso 78. "
            "Pesa como 70 kilos y mide 158 centímetros."
        ),
        "esperado": {
            "edad": 55, "presion_sistolica": 170,
            "presion_diastolica": 100, "glucosa_mg_dl": 95,
        },
    },
]


# ── 1. Healthcheck ─────────────────────────────────────────────────────────────
def test_health():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "isolation_forest" in data["modelos"]


# ── 2. /extraccion ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("caso", CASOS, ids=[c["nombre"] for c in CASOS])
def test_extraccion(caso):
    r = httpx.post(f"{BASE_URL}/extraccion", json={"texto": caso["texto"]})
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    data = r.json()
    assert "inferencia_id" in data
    campos = data["campos_extraidos"]
    for campo, valor_esperado in caso["esperado"].items():
        assert campos.get(campo) == valor_esperado, (
            f"{caso['nombre']} | {campo}: esperado={valor_esperado}, obtenido={campos.get(campo)}"
        )


# ── 3. /deteccion-anomalias — paciente normal ──────────────────────────────────
def test_deteccion_normal():
    r = httpx.post(f"{BASE_URL}/deteccion-anomalias", json={
        "edad": 34, "peso_kg": 72.4, "talla_cm": 151.4,
        "presion_sistolica": 118, "presion_diastolica": 75,
        "glucosa_mg_dl": 87, "temperatura_c": 37.0,
        "frecuencia_cardiaca_bpm": 77, "duracion_sintomas_dias": 3,
        "categoria_sintoma": "Respiratorio",
    })
    assert r.status_code == 200
    data = r.json()
    assert "es_anomalia" in data
    assert "score" in data
    assert data["nivel_riesgo"] in ("normal", "sospechoso", "anomalo")


# ── 4. /deteccion-anomalias — paciente anómalo (glucosa 341, fiebre, FC alta) ──
def test_deteccion_anomalo():
    r = httpx.post(f"{BASE_URL}/deteccion-anomalias", json={
        "edad": 63, "peso_kg": 68.8, "talla_cm": 149.4,
        "presion_sistolica": 165, "presion_diastolica": 83,
        "glucosa_mg_dl": 341, "temperatura_c": 39.0,
        "frecuencia_cardiaca_bpm": 113, "duracion_sintomas_dias": 1,
        "categoria_sintoma": "Respiratorio",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["es_anomalia"] is True
    assert data["score"] < 0


# ── 5. /consulta-completa — los 5 estilos ──────────────────────────────────────
@pytest.mark.parametrize("caso", CASOS, ids=[c["nombre"] for c in CASOS])
def test_consulta_completa(caso):
    r = httpx.post(f"{BASE_URL}/consulta-completa", json={"texto": caso["texto"]})
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    data = r.json()
    assert "inferencia_id" in data
    assert "extraccion" in data
    campos = data["extraccion"]
    for campo, valor_esperado in caso["esperado"].items():
        assert campos.get(campo) == valor_esperado, (
            f"{caso['nombre']} | {campo}: esperado={valor_esperado}, obtenido={campos.get(campo)}"
        )
    # Si se extrajo todo, debe haber resultado de anomalía
    if not data.get("advertencia"):
        assert data["anomalia"] is not None
        assert "es_anomalia" in data["anomalia"]


# ── 6. GET /inferencias — verificar persistencia ───────────────────────────────
def test_inferencias_persistidas():
    r = httpx.get(f"{BASE_URL}/inferencias", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1
    tipos_validos = {"extraccion", "anomalia", "completa"}
    for inf in data["inferencias"]:
        assert inf["tipo"] in tipos_validos


def test_inferencias_filtro_tipo():
    r = httpx.get(f"{BASE_URL}/inferencias", params={"tipo": "completa", "limit": 5})
    assert r.status_code == 200
    data = r.json()
    for inf in data["inferencias"]:
        assert inf["tipo"] == "completa"
