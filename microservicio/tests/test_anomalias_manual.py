"""
Prueba manual del endpoint POST /deteccion-anomalias.

Flujo:
  1. Login en http://localhost:8000/auth/login  →  obtiene JWT
  2. Manda 6 casos al ML microservicio en http://localhost:8001/deteccion-anomalias
  3. Imprime resultado por caso y resumen final

Ejecutar:
    python tests/test_anomalias_manual.py
"""

import sys

try:
    import httpx as _http_lib
    _USE_HTTPX = True
except ImportError:
    try:
        import requests as _http_lib
        _USE_HTTPX = False
    except ImportError:
        sys.exit("ERROR: instala httpx o requests antes de correr este script.\n"
                 "  pip install httpx")

AUTH_URL = "http://localhost:8000/auth/login"
BASE_URL = "http://localhost:8001"

# ── Umbrales del IsolationForestAdapter (para el aviso de diagnóstico) ─────────
_UMBRAL_SOSPECHOSO = -0.05

# ── Helpers HTTP ───────────────────────────────────────────────────────────────

def _post(url: str, json: dict, headers: dict | None = None) -> dict:
    if _USE_HTTPX:
        r = _http_lib.post(url, json=json, headers=headers or {}, timeout=15)
    else:
        r = _http_lib.post(url, json=json, headers=headers or {}, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Paso 1: Login ──────────────────────────────────────────────────────────────

def obtener_token() -> str:
    print("Autenticando...", end=" ")
    try:
        data = _post(AUTH_URL, {"correo": "admin@epidiagnostix.mx", "contrasena": "Admin2024!"})
    except Exception as exc:
        print(f"\nERROR al hacer login en {AUTH_URL}: {exc}")
        print("  Verifica que el servicio de autenticación esté corriendo en el puerto 8000.")
        sys.exit(1)

    token = data.get("access_token")
    if not token:
        print(f"\nERROR: la respuesta no contiene 'access_token'. Respuesta: {data}")
        sys.exit(1)
    print("OK\n")
    return token


# ── Casos de prueba ────────────────────────────────────────────────────────────
#   nivel_esperado puede ser un string exacto O una lista de strings aceptables

CASOS = [
    {
        "nombre": "CASO 1 — Paciente normal",
        "payload": {
            "edad": 35, "peso_kg": 70.0, "talla_cm": 168.0,
            "presion_sistolica": 118, "presion_diastolica": 76,
            "glucosa_mg_dl": 95, "temperatura_c": 36.6,
            "frecuencia_cardiaca_bpm": 72, "duracion_sintomas_dias": 2,
            "categoria_sintoma": "Respiratorio",
        },
        "nivel_esperado": "normal",
    },
    {
        "nombre": "CASO 2 — Fiebre alta con taquicardia",
        "payload": {
            "edad": 28, "peso_kg": 65.0, "talla_cm": 162.0,
            "presion_sistolica": 130, "presion_diastolica": 85,
            "glucosa_mg_dl": 98, "temperatura_c": 39.8,
            "frecuencia_cardiaca_bpm": 118, "duracion_sintomas_dias": 3,
            "categoria_sintoma": "Gastrointestinal",
        },
        "nivel_esperado": ["sospechoso", "anomalo"],
    },
    {
        "nombre": "CASO 3 — Crisis hipertensiva",
        "payload": {
            "edad": 58, "peso_kg": 92.0, "talla_cm": 170.0,
            "presion_sistolica": 195, "presion_diastolica": 125,
            "glucosa_mg_dl": 142, "temperatura_c": 37.1,
            "frecuencia_cardiaca_bpm": 95, "duracion_sintomas_dias": 1,
            "categoria_sintoma": "Hipertension",
        },
        "nivel_esperado": "anomalo",
    },
    {
        "nombre": "CASO 4 — Hipoglucemia severa",
        "payload": {
            "edad": 67, "peso_kg": 58.0, "talla_cm": 155.0,
            "presion_sistolica": 88, "presion_diastolica": 52,
            "glucosa_mg_dl": 38, "temperatura_c": 36.2,
            "frecuencia_cardiaca_bpm": 108, "duracion_sintomas_dias": 1,
            "categoria_sintoma": "Diabetes",
        },
        "nivel_esperado": ["sospechoso", "anomalo"],
    },
    {
        "nombre": "CASO 5 — Pediátrico normal",
        "payload": {
            "edad": 4, "peso_kg": 16.0, "talla_cm": 102.0,
            "presion_sistolica": 95, "presion_diastolica": 60,
            "glucosa_mg_dl": 88, "temperatura_c": 37.0,
            "frecuencia_cardiaca_bpm": 100, "duracion_sintomas_dias": 1,
            "categoria_sintoma": "Vacunacion",
        },
        "nivel_esperado": "normal",
    },
    {
        "nombre": "CASO 6 — Combinación extrema múltiple",
        "payload": {
            "edad": 45, "peso_kg": 120.0, "talla_cm": 165.0,
            "presion_sistolica": 188, "presion_diastolica": 118,
            "glucosa_mg_dl": 380, "temperatura_c": 39.5,
            "frecuencia_cardiaca_bpm": 130, "duracion_sintomas_dias": 5,
            "categoria_sintoma": "Diabetes",
        },
        "nivel_esperado": "anomalo",
    },
]


# ── Paso 2: Ejecutar casos ─────────────────────────────────────────────────────

def _es_correcto(nivel_obtenido: str, nivel_esperado) -> bool:
    if isinstance(nivel_esperado, list):
        return nivel_obtenido in nivel_esperado
    return nivel_obtenido == nivel_esperado


def _label_esperado(nivel_esperado) -> str:
    if isinstance(nivel_esperado, list):
        return " o ".join(nivel_esperado)
    return nivel_esperado


def correr_casos(token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/deteccion-anomalias"
    resultados = []

    for caso in CASOS:
        print(f"  {caso['nombre']}")
        try:
            resp = _post(url, caso["payload"], headers)
        except Exception as exc:
            print(f"    ERROR HTTP: {exc}\n")
            resultados.append({"caso": caso, "error": str(exc)})
            continue

        nivel      = resp.get("nivel_riesgo", "?")
        score      = resp.get("score", float("nan"))
        es_anomalia = resp.get("es_anomalia", None)
        correcto   = _es_correcto(nivel, caso["nivel_esperado"])

        marca = "✓" if correcto else "✗"
        estado = "" if correcto else "  ← REVISAR"
        print(f"    {marca}  nivel_riesgo={nivel!r}  score={score:+.4f}  "
              f"es_anomalia={es_anomalia}  (esperado: {_label_esperado(caso['nivel_esperado'])}){estado}")
        print()

        resultados.append({
            "caso": caso,
            "nivel": nivel,
            "score": score,
            "es_anomalia": es_anomalia,
            "correcto": correcto,
        })

    return resultados


# ── Paso 3: Resumen ────────────────────────────────────────────────────────────

def imprimir_resumen(resultados: list[dict]) -> None:
    exitosos = [r for r in resultados if r.get("correcto")]
    errores  = [r for r in resultados if "error" in r]
    total    = len(resultados)
    print("=" * 60)
    print(f"Casos correctos: {len(exitosos)}/{total}")

    if errores:
        print(f"Casos con error HTTP: {len(errores)}")
        for r in errores:
            print(f"  - {r['caso']['nombre']}: {r['error']}")

    # Scores promedio por nivel real obtenido
    scores_por_nivel: dict[str, list[float]] = {}
    for r in resultados:
        if "score" in r and r["score"] == r["score"]:   # excluye nan
            nivel = r.get("nivel", "?")
            scores_por_nivel.setdefault(nivel, []).append(r["score"])

    if scores_por_nivel:
        promedios = {n: sum(v) / len(v) for n, v in scores_por_nivel.items()}
        partes = []
        for nivel in ("normal", "sospechoso", "anomalo"):
            if nivel in promedios:
                partes.append(f"{nivel}={promedios[nivel]:+.4f}")
        if partes:
            print(f"Scores promedio por nivel: {', '.join(partes)}")

    # ── Aviso si normal↔anomalo se confunden ───────────────────────────────────
    confusiones = []
    for r in resultados:
        if "nivel" not in r:
            continue
        esperado = r["caso"]["nivel_esperado"]
        obtenido = r["nivel"]
        # Solo alerta en confusión grave: normal↔anomalo (no sospechoso)
        if (esperado == "normal" and obtenido == "anomalo") or \
           (esperado == "anomalo" and obtenido == "normal") or \
           (isinstance(esperado, list) and "normal" not in esperado and obtenido == "normal"):
            confusiones.append(r["caso"]["nombre"])

    if confusiones:
        print()
        print("⚠ POSIBLE PROBLEMA EN EL MODELO — revisar contamination o threshold "
              "del IsolationForestAdapter")
        print(f"  Umbrales actuales: sospechoso > {_UMBRAL_SOSPECHOSO}")
        print("  Casos con confusión grave:")
        for nombre in confusiones:
            print(f"    - {nombre}")

    print("=" * 60)


# ── Punto de entrada ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    lib_name = "httpx" if _USE_HTTPX else "requests"
    print(f"EpiDiagnostix — Test de anomalías (usando {lib_name})")
    print(f"Auth:    {AUTH_URL}")
    print(f"ML API:  {BASE_URL}")
    print()

    token = obtener_token()

    print("Corriendo 6 casos de prueba...\n")
    resultados = correr_casos(token)

    print()
    imprimir_resumen(resultados)
