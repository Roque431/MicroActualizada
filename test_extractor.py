"""
Test rápido del extractor NER (spaCy + regex).
Corre desde la raíz del proyecto:
    python test_extractor.py
"""
import re
import unicodedata
import joblib
from dataclasses import dataclass
from typing import Optional


def _norm(s):
    """Quita tildes y pasa a minúsculas para comparaciones robustas."""
    return unicodedata.normalize('NFD', str(s)).encode('ascii', 'ignore').decode().lower().strip()

# ── Cargar configuración guardada ─────────────────────────────────────────────
cfg = joblib.load("models/extractor_ner_config.joblib")
CATEGORIA_KEYWORDS = cfg["categoria_keywords"]
SEXO_MAP           = cfg["sexo_map"]

# ── Patrones regex (mismos del notebook) ──────────────────────────────────────
RE_EDAD = [
    re.compile(r'(?:edad[:\s]+)(\d{1,3})\s*a[ñn]os?', re.I),
    re.compile(r'(?:paciente\s+(?:masculino|femenino|hombre|mujer|var[oó]n|femenina)\s+)?(?:de\s+)(\d{1,3})\s*a[ñn]os?', re.I),
    re.compile(r'(\d{1,3})\s*a[ñn]os?\s+(?:de\s+edad|cumplidos?)', re.I),
    re.compile(r'(\d{1,3})\s*a[ñn]os?', re.I),
]
RE_SEXO = re.compile(r'\b(masculino|femenino|hombre|mujer|var[oó]n|femenina|ni[ñn]o|ni[ñn]a|se[ñn]ora|se[ñn]or)\b', re.I)
SEXO_MAP_EXTRA = {'señora': 'F', 'senora': 'F', 'señor': 'M', 'senor': 'M'}
RE_PA   = re.compile(
    r'(?:presi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|tensi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|PA[:\s]+)'
    r'(\d{2,3})\s*[/\-sobre\s]+(\d{2,3})\s*(?:mmHg)?', re.I)
RE_PA2  = re.compile(r'(\d{2,3})\s+sobre\s+(\d{2,3})\s*(?:mmHg)?', re.I)
RE_GLUCOSA = [
    re.compile(r'glucos[ao]\s+(?:de\s+|capilar\s+)?(\d{2,3})\s*(?:mg/d[lL]|miligramos?)?', re.I),
    re.compile(r'glucemia\s+(?:capilar\s+)?(\d{2,3})\s*(?:mg/d[lL])?', re.I),
    re.compile(r'az[uú]car\s+en\s+sangre\s+(\d{2,3})', re.I),
    re.compile(r'az[uú]car\s+\w+\s+\w+,?\s*(\d{2,3})', re.I),
    re.compile(r'nivel\s+de\s+glucos[ao]\s+(\d{2,3})', re.I),
]
RE_TEMP = [
    re.compile(r'temperatura\s+(?:corporal\s+|de\s+)?(\d{2})(?:[.,](\d))\s*(?:grados?|[°ºC])', re.I),
    re.compile(r'T[:\s]+?(\d{2})(?:[.,](\d))?\s*grados?', re.I),
    re.compile(r'(\d{2})(?:[.,](\d))?[°º]C', re.I),
    re.compile(r'temperatura\s+\w*\s*(\d{2})(?:[.,](\d))?', re.I),
]
RE_FC = [
    re.compile(r'frecuencia\s+cardiaca\s+(?:de\s+)?(\d{2,3})\s*(?:latidos?\s+(?:por\s+)?minuto|lpm|por\s+minuto)', re.I),
    re.compile(r'pulso\s+(\d{2,3})\s*(?:lpm|por\s+minuto)?', re.I),
    re.compile(r'FC[:\s]+?(\d{2,3})\s*(?:lpm)?', re.I),
    re.compile(r'ritmo\s+cardiaco\s+(\d{2,3})', re.I),
]
RE_PESO = [
    re.compile(r'peso\s+(?:corporal[:\s]+)?(\d{2,3}(?:[.,]\d)?)\s*(?:kilogramos?|kg)', re.I),
    re.compile(r'pesa\s+(\d{2,3}(?:[.,]\d)?)\s*(?:kg|kilos?)', re.I),
    re.compile(r'(\d{2,3}(?:[.,]\d)?)\s*kg\s+de\s+peso', re.I),
    re.compile(r'(\d{2,3}(?:[.,]\d)?)\s*kilos?\b', re.I),
]
RE_TALLA = [
    re.compile(r'talla\s+(\d{2,3}(?:[.,]\d)?)\s*(?:cent[ií]metros?|cm)', re.I),
    re.compile(r'estatura\s+(?:de\s+)?(\d{2,3}(?:[.,]\d)?)\s*(?:cm|cent[ií]metros?)', re.I),
    re.compile(r'mide\s+(\d{2,3}(?:[.,]\d)?)\s*cm', re.I),
    re.compile(r'altura\s+(\d{2,3}(?:[.,]\d)?)\s*(?:cm|cent[ií]metros?)', re.I),
    re.compile(r'(\d{2,3}(?:[.,]\d)?)\s*(?:cm|cent[ií]metros?)\b', re.I),
]
RE_DUR = [
    re.compile(r'(\d{1,3})\s*d[ií]as?\s+de\s+evoluci[oó]n', re.I),
    re.compile(r'desde\s+hace\s+(\d{1,3})\s*d[ií]as?', re.I),
    re.compile(r'cuadro\s+(?:cl[ií]nico\s+)?de\s+(\d{1,3})\s*d[ií]as?', re.I),
    re.compile(r'lleva\s+(\d{1,3})\s*d[ií]as?', re.I),
    re.compile(r'evoluci[oó]n\s+de\s+(\d{1,3})\s*d[ií]as?', re.I),
    re.compile(r'(\d{1,3})\s*d[ií]as?', re.I),
]


# ── Funciones auxiliares ──────────────────────────────────────────────────────
def _first_num(patterns, text):
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).replace(',', '.')
    return None

def _parse_temp(text):
    for pat in RE_TEMP:
        m = pat.search(text)
        if m:
            entera  = m.group(1)
            decimal = m.group(2) if m.lastindex >= 2 and m.group(2) else '0'
            return float(f'{entera}.{decimal}')
    return None

def _parse_pa(text):
    for pat in [RE_PA, RE_PA2]:
        m = pat.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None

def _parse_categoria(text):
    text_lower = text.lower()
    scores = {cat: sum(kw.lower() in text_lower for kw in kws)
              for cat, kws in CATEGORIA_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── Pipeline principal ────────────────────────────────────────────────────────
def extraer(texto: str) -> dict:
    ps, pd_ = _parse_pa(texto)
    gl  = _first_num(RE_GLUCOSA, texto)
    tc  = _parse_temp(texto)
    fc  = _first_num(RE_FC, texto)
    pk  = _first_num(RE_PESO, texto)
    tl  = _first_num(RE_TALLA, texto)
    dur = _first_num(RE_DUR, texto)

    edad = None
    for pat in RE_EDAD:
        m = pat.search(texto)
        if m:
            c = int(m.group(1))
            if 0 < c <= 120:
                edad = c
                break

    m_sexo = RE_SEXO.search(texto)
    if m_sexo:
        k = m_sexo.group(1).lower()
        sexo = SEXO_MAP.get(k) or SEXO_MAP_EXTRA.get(k)
    else:
        sexo = None

    return {
        "edad":                    edad,
        "sexo":                    sexo,
        "peso_kg":                 round(float(pk), 1) if pk else None,
        "talla_cm":                round(float(tl), 1) if tl else None,
        "presion_sistolica":       ps,
        "presion_diastolica":      pd_,
        "glucosa_mg_dl":           int(float(gl)) if gl else None,
        "temperatura_c":           tc,
        "frecuencia_cardiaca_bpm": int(float(fc)) if fc else None,
        "duracion_sintomas_dias":  int(float(dur)) if dur else None,
        "categoria_sintoma":       _parse_categoria(texto),
    }


# ── Casos de prueba ───────────────────────────────────────────────────────────
CASOS = [
    {
        "descripcion": "Caso 1 — dictado estilo SOAP",
        "texto": (
            "Consulta médica. Paciente masculino de 45 años de edad. "
            "Peso 82.0 kg, talla 170.5 cm. "
            "Motivo: dolor abdominal, diarrea, deshidratación, 5 días de evolución. "
            "Signos vitales: presión arterial de 125/82 mmHg, glucosa de 98 mg/dl, "
            "temperatura 37.2°C, frecuencia cardiaca de 91 latidos por minuto."
        ),
        "esperado": {
            "edad": 45, "sexo": "M", "peso_kg": 82.0, "talla_cm": 170.5,
            "presion_sistolica": 125, "presion_diastolica": 82,
            "glucosa_mg_dl": 98, "temperatura_c": 37.2,
            "frecuencia_cardiaca_bpm": 91, "duracion_sintomas_dias": 5,
            "categoria_sintoma": "Gastrointestinal"
        }
    },
    {
        "descripcion": "Caso 2 — dictado rapido abreviado",
        "texto": (
            "Paciente femenina, 67 años. PA: 160/95, glucemia capilar 310 mg/dL. "
            "T: 36.8 grados. FC: 88 lpm. Peso 63.5 kg, estatura de 155.0 cm. "
            "Síntomas: fatiga y pérdida de peso desde hace 3 días."
        ),
        "esperado": {
            "edad": 67, "sexo": "F", "peso_kg": 63.5, "talla_cm": 155.0,
            "presion_sistolica": 160, "presion_diastolica": 95,
            "glucosa_mg_dl": 310, "temperatura_c": 36.8,
            "frecuencia_cardiaca_bpm": 88, "duracion_sintomas_dias": 3,
            "categoria_sintoma": "Diabetes"
        }
    },
    {
        "descripcion": "Caso 3 — estilo coloquial comunidad",
        "texto": (
            "Se atiende a mujer de 32 años que lleva 7 días con tos y dificultad para respirar. "
            "Presión 118 sobre 74, azúcar en sangre 89, temperatura 38.5, pulso 102. "
            "Pesa 58 kilos, mide 162 cm."
        ),
        "esperado": {
            "edad": 32, "sexo": "F", "peso_kg": 58.0, "talla_cm": 162.0,
            "presion_sistolica": 118, "presion_diastolica": 74,
            "glucosa_mg_dl": 89, "temperatura_c": 38.5,
            "frecuencia_cardiaca_bpm": 102, "duracion_sintomas_dias": 7,
            "categoria_sintoma": "Respiratorio"
        }
    },
    {
        "descripcion": "Caso 4 — nino con vacunacion",
        "texto": (
            "Paciente masculino, edad: 2 años. Consulta por fiebre leve postvacuna "
            "y control de esquema de vacunacion con 1 dia de evolucion. "
            "Peso corporal: 12.5 kg, talla 82.0 centimetros. "
            "Tension arterial 100/65 mmHg, temperatura de 37.8 grados, "
            "frecuencia cardiaca de 110 lpm, glucosa 88 mg/dl."
        ),
        "esperado": {
            "edad": 2, "sexo": "M", "peso_kg": 12.5, "talla_cm": 82.0,
            "presion_sistolica": 100, "presion_diastolica": 65,
            "glucosa_mg_dl": 88, "temperatura_c": 37.8,
            "frecuencia_cardiaca_bpm": 110, "duracion_sintomas_dias": 1,
            "categoria_sintoma": "Vacunacion"
        }
    },
    {
        "descripcion": "Caso 5 — texto libre sin estructura (prueba de robustez)",
        "texto": (
            "La señora tiene como 55 años, es hipertensa, refiere zumbido en los oídos "
            "y visión borrosa desde hace 4 días. Le tomé la presión y salió 170 sobre 100. "
            "Su azúcar estaba bien, 95. Temperatura normal 36.5. Pulso 78. "
            "Pesa como 70 kilos y mide 158 centímetros."
        ),
        "esperado": {
            "edad": 55, "sexo": "F", "peso_kg": 70.0, "talla_cm": 158.0,
            "presion_sistolica": 170, "presion_diastolica": 100,
            "glucosa_mg_dl": 95, "temperatura_c": 36.5,
            "frecuencia_cardiaca_bpm": 78, "duracion_sintomas_dias": 4,
            "categoria_sintoma": "Hipertension"
        }
    },
]


# ── Ejecutar pruebas ──────────────────────────────────────────────────────────
CAMPOS = [
    "edad", "sexo", "peso_kg", "talla_cm",
    "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c",
    "frecuencia_cardiaca_bpm", "duracion_sintomas_dias",
    "categoria_sintoma",
]

TOLERANCIAS = {
    "edad": 1, "peso_kg": 1.0, "talla_cm": 1.0,
    "presion_sistolica": 1, "presion_diastolica": 1,
    "glucosa_mg_dl": 1, "temperatura_c": 0.2,
    "frecuencia_cardiaca_bpm": 1, "duracion_sintomas_dias": 1,
}

total_campos = 0
total_ok     = 0

for caso in CASOS:
    print("\n" + "=" * 60)
    print(f"  {caso['descripcion']}")
    print("=" * 60)
    print(f"  TEXTO: {caso['texto'][:90]}...")
    print()

    resultado  = extraer(caso["texto"])
    esperado   = caso["esperado"]
    caso_ok    = 0
    caso_total = 0

    print(f"  {'Campo':<28} {'Extraido':>15}  {'Esperado':>15}  OK?")
    print("  " + "-" * 62)

    for campo in CAMPOS:
        pred  = resultado.get(campo)
        real  = esperado.get(campo)
        tol   = TOLERANCIAS.get(campo)

        if real is None:
            continue

        if tol is not None and pred is not None:
            ok = abs(float(pred) - float(real)) <= tol
        else:
            ok = _norm(pred) == _norm(real)

        marca = "OK" if ok else "FALLA"
        print(f"  {campo:<28} {str(pred):>15}  {str(real):>15}  {marca}")

        caso_total  += 1
        total_campos += 1
        if ok:
            caso_ok    += 1
            total_ok   += 1

    pct = caso_ok / caso_total * 100 if caso_total else 0
    print(f"\n  Resultado: {caso_ok}/{caso_total} campos correctos ({pct:.0f}%)")

print("\n" + "=" * 60)
pct_global = total_ok / total_campos * 100 if total_campos else 0
print(f"  RESULTADO GLOBAL: {total_ok}/{total_campos} ({pct_global:.1f}% exactitud)")
print("=" * 60)
