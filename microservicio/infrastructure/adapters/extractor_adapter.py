"""
Adapter que implementa ExtractorPort reutilizando la lógica de reglas/regex
del extractor ya validado (test_extractor.py). No se reescribe la lógica,
solo se adapta al puerto hexagonal.
"""
import re
import unicodedata

import joblib

from domain.entities import ConsultaClinica, ResultadoExtraccion
from domain.ports import ExtractorPort

# ── Patrones regex ─────────────────────────────────────────────────────────────
RE_EDAD = [
    re.compile(r'(?:edad[:\s]+)(\d{1,3})\s*a[ñn]os?', re.I),
    re.compile(r'(?:paciente\s+(?:masculino|femenino|hombre|mujer|var[oó]n|femenina)\s+)?(?:de\s+)(\d{1,3})\s*a[ñn]os?', re.I),
    re.compile(r'(\d{1,3})\s*a[ñn]os?\s+(?:de\s+edad|cumplidos?)', re.I),
    re.compile(r'(\d{1,3})\s*a[ñn]os?', re.I),
]
RE_SEXO = re.compile(
    r'\b(masculino|femenino|hombre|mujer|var[oó]n|femenina|ni[ñn]o|ni[ñn]a|se[ñn]ora|se[ñn]or)\b',
    re.I,
)
SEXO_MAP = {
    'masculino': 'M', 'hombre': 'M', 'varón': 'M', 'varon': 'M', 'niño': 'M',
    'señor': 'M', 'senor': 'M',
    'femenino': 'F', 'mujer': 'F', 'femenina': 'F', 'niña': 'F',
    'señora': 'F', 'senora': 'F',
}
RE_PA  = re.compile(
    r'(?:presi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|tensi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|PA[:\s]+)'
    r'(\d{2,3})\s*[/\-sobre\s]+(\d{2,3})\s*(?:mmHg)?', re.I)
RE_PA2 = re.compile(r'(\d{2,3})\s+sobre\s+(\d{2,3})\s*(?:mmHg)?', re.I)
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
CATEGORIA_KEYWORDS = {
    'Respiratorio':    ['respiratorio','respiratoria','tos','dificultad para respirar',
                        'congestión nasal','bronquitis','neumonía','gripe'],
    'Gastrointestinal':['gastrointestinal','diarrea','vómito','náusea','náuseas',
                        'dolor abdominal','gastroenteritis','deshidratación'],
    'Hipertensión':    ['hipertensión','hipertenso','hipertensa','tensión alta',
                        'presión alta','visión borrosa','zumbido en oídos'],
    'Diabetes':        ['diabetes','diabético','diabética','glucemia alta',
                        'azúcar alta','hiperglucemia'],
    'Vacunación':      ['vacunación','vacuna','esquema de vacunación',
                        'fiebre postvacuna'],
    'Nutrición':       ['nutrición','nutricional','palidez','debilidad',
                        'desnutrición','retraso en talla','anemia'],
    'Embarazo':        ['embarazo','embarazada','gestación','prenatal'],
    'Traumatología':   ['traumatología','traumatológico','esguince',
                        'fractura','herida','golpe','contusión'],
    'Dermatológico':   ['dermatológico','dermatológica','piel','erupción',
                        'sarpullido','dermatitis','urticaria'],
}

TODOS_LOS_CAMPOS = [
    "edad", "sexo", "peso_kg", "talla_cm",
    "presion_sistolica", "presion_diastolica",
    "glucosa_mg_dl", "temperatura_c",
    "frecuencia_cardiaca_bpm", "duracion_sintomas_dias",
    "categoria_sintoma",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
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
    tl = text.lower()
    scores = {cat: sum(kw.lower() in tl for kw in kws)
              for cat, kws in CATEGORIA_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ── Adapter ────────────────────────────────────────────────────────────────────
class ExtractorAdapter(ExtractorPort):
    """
    Implementa ExtractorPort usando el motor de reglas/regex entrenado.
    Se instancia una sola vez en el arranque (singleton vía FastAPI lifespan).
    """

    def extraer(self, texto: str) -> ResultadoExtraccion:
        ps, pd_ = _parse_pa(texto)
        gl      = _first_num(RE_GLUCOSA, texto)
        tc      = _parse_temp(texto)
        fc      = _first_num(RE_FC, texto)
        pk      = _first_num(RE_PESO, texto)
        tl      = _first_num(RE_TALLA, texto)
        dur     = _first_num(RE_DUR, texto)

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
            k = unicodedata.normalize('NFC', m_sexo.group(1).lower())
            sexo = SEXO_MAP.get(k)
        else:
            sexo = None

        campos = ConsultaClinica(
            edad=edad,
            sexo=sexo,
            peso_kg=round(float(pk), 1) if pk else None,
            talla_cm=round(float(tl), 1) if tl else None,
            presion_sistolica=ps,
            presion_diastolica=pd_,
            glucosa_mg_dl=int(float(gl)) if gl else None,
            temperatura_c=tc,
            frecuencia_cardiaca_bpm=int(float(fc)) if fc else None,
            duracion_sintomas_dias=int(float(dur)) if dur else None,
            categoria_sintoma=_parse_categoria(texto),
        )

        no_extraidos = [
            f for f in TODOS_LOS_CAMPOS
            if getattr(campos, f) is None
        ]

        return ResultadoExtraccion(campos=campos, campos_no_extraidos=no_extraidos)
