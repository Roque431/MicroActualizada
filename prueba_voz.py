"""
Prueba de voz para EpiDiagnostix-Mayab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graba la consulta médico-paciente, transcribe el audio,
guarda el texto completo y muestra SOLO las variables clínicas extraídas.

Uso:
    python prueba_voz.py
    python prueba_voz.py --whisper      (modo offline, sin internet)
    python prueba_voz.py --api          (envía al microservicio Docker)
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

# ── Configuración de audio ──────────────────────────────────────────────────
SAMPLE_RATE  = 16_000   # 16 kHz — estándar para reconocimiento de voz
CHANNELS     = 1
DTYPE        = 'int16'
SESIONES_DIR = Path("sesiones_voz")
SESIONES_DIR.mkdir(exist_ok=True)

# ── Colores ANSI para la terminal ───────────────────────────────────────────
ROJO    = "\033[91m"
VERDE   = "\033[92m"
AMARILLO= "\033[93m"
AZUL    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
GRIS    = "\033[90m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

# ── Estado global de grabación ──────────────────────────────────────────────
_grabando   = False
_fragmentos = []


def _callback_audio(indata, frames, time_info, status):
    """Callback de sounddevice — acumula fragmentos de audio."""
    if _grabando:
        _fragmentos.append(indata.copy())


def _hilo_indicador():
    """Muestra un contador de tiempo mientras graba."""
    inicio = time.time()
    while _grabando:
        transcurrido = int(time.time() - inicio)
        mins = transcurrido // 60
        segs = transcurrido % 60
        print(f"\r  {ROJO}● REC{RESET}  {BOLD}{mins:02d}:{segs:02d}{RESET}  "
              f"{GRIS}(ENTER para detener){RESET}   ", end="", flush=True)
        time.sleep(0.5)


def grabar_hasta_enter() -> np.ndarray:
    """Graba desde el micrófono hasta que el usuario presiona ENTER."""
    global _grabando, _fragmentos
    _fragmentos = []
    _grabando   = True

    hilo = threading.Thread(target=_hilo_indicador, daemon=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=_callback_audio):
        hilo.start()
        input()          # ← bloquea hasta ENTER

    _grabando = False
    time.sleep(0.1)      # deja que el hilo indicador termine
    print()

    if not _fragmentos:
        return np.array([], dtype=np.int16)
    return np.concatenate(_fragmentos, axis=0).flatten()


def guardar_wav(audio: np.ndarray, ruta: str):
    """Guarda el array de audio como WAV."""
    with wave.open(ruta, 'w') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def transcribir_google(wav_path: str) -> str:
    """Transcripción con Google Speech (necesita internet, gratuito)."""
    import speech_recognition as sr
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True

    with sr.AudioFile(wav_path) as source:
        # Chunk de hasta 60 s para no superar el límite de Google
        audio = r.record(source)

    try:
        texto = r.recognize_google(audio, language='es-MX')
        return texto
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"\n{ROJO}Google Speech no disponible: {e}{RESET}")
        print(f"{AMARILLO}Prueba con --whisper para modo offline.{RESET}")
        return ""


def transcribir_whisper(wav_path: str) -> str:
    """
    Transcripción con Whisper (100% offline).
    Primera vez descarga el modelo 'small' (~244 MB).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(f"{ROJO}faster-whisper no instalado. Corre: pip install faster-whisper{RESET}")
        return ""

    print(f"  {GRIS}Cargando modelo Whisper (primera vez descarga ~244 MB)...{RESET}")
    modelo = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = modelo.transcribe(wav_path, language="es", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


# ── Extractor de variables clínicas (reutiliza el módulo ya validado) ───────
def extraer_variables_local(texto: str) -> dict:
    """Extrae variables clínicas usando el motor de reglas (sin API)."""
    import re, unicodedata

    SEXO_MAP = {
        'masculino': 'M', 'hombre': 'M', 'varon': 'M', 'varón': 'M', 'niño': 'M',
        'senor': 'M', 'señor': 'M',
        'femenino': 'F', 'mujer': 'F', 'femenina': 'F', 'niña': 'F',
        'senora': 'F', 'señora': 'F',
    }
    CATEGORIA_KEYWORDS = {
        'Respiratorio':    ['respiratorio','tos','dificultad para respirar','congestion','bronquitis','gripe','influenza','neumonia','bronconeumonia'],
        'Gastrointestinal':['gastrointestinal','diarrea','vomito','nausea','dolor abdominal','deshidratacion','gastroenteritis'],
        'Hipertension':    ['hipertension','hipertenso','hipertensa','presion alta','vision borrosa','zumbido'],
        'Diabetes':        ['diabetes','diabetico','glucemia alta','azucar alta','hiperglucemia'],
        'Vacunacion':      ['vacunacion','vacuna','esquema de vacunacion','postvacuna'],
        'Nutricion':       ['nutricion','palidez','desnutricion','anemia','debilidad','retraso en talla'],
        'Embarazo':        ['embarazo','embarazada','gestacion','prenatal'],
        'Traumatologia':   ['traumatologia','esguince','fractura','herida','contusion','golpe'],
        'Dermatologico':   ['dermatologico','dermatitis','erupcion','sarpullido','urticaria'],
        # FIX: enfermedades vectoriales/infecciosas no estaban
        'Infeccioso/Vectorial': ['dengue','sospechoso de dengue','fiebre dengue','dengue hemorragico',
                                 'chikungunya','zika','paludismo','malaria','leptospirosis',
                                 'fiebre tifoidea','rickettsia','infeccioso','infecciosa'],
    }

    def _first(patterns, txt):
        for p in patterns:
            m = re.search(p, txt, re.I)
            if m:
                return m.group(1).replace(',','.')
        return None

    def _pa(txt):
        for p in [
            r'(?:presi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|tensi[oó]n\s+(?:arterial\s+)?(?:de\s+)?|PA[:\s]+|TA[:\s]+)(\d{2,3})\s*[/\-]\s*(\d{2,3})',
            r'(\d{2,3})\s+sobre\s+(\d{2,3})',
        ]:
            m = re.search(p, txt, re.I)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def _temp(txt):
        for p in [
            r'temperatura\s+(?:corporal\s+|de\s+)?(\d{2})(?:[.,](\d))\s*(?:grados?|[°ºC])',
            r'T[:\s]+?(\d{2})(?:[.,](\d))?\s*grados?',
            r'(\d{2})(?:[.,](\d))?[°º]C',
            r'temperatura\s+\w*\s*(\d{2})(?:[.,](\d))?',
            r'fiebre\s+(?:de\s+)?(\d{2})(?:[.,](\d))?',
        ]:
            m = re.search(p, txt, re.I)
            if m:
                dec = m.group(2) if m.lastindex >= 2 and m.group(2) else '0'
                return float(f'{m.group(1)}.{dec}')
        return None

    def _norm(s):
        return unicodedata.normalize('NFD', str(s)).encode('ascii','ignore').decode().lower()

    # FIX GLOBAL: normalizar texto quitando tildes antes de aplicar regex.
    # Esto resuelve "cardíaca" vs "cardiaca", "presión" vs "presion", etc.
    # de un solo golpe sin tener que actualizar cada patrón individualmente.
    txt      = _norm(texto)   # sin tildes — para regex
    txt_orig = texto          # con tildes  — para categoría keywords

    # Edad
    edad = None
    for p in [
        r'(?:edad[:\s]+)(\d{1,3})\s*a[ñn]os?',
        r'(?:de\s+)(\d{1,3})\s*a[ñn]os?',
        r'(\d{1,3})\s*a[ñn]os?\s+(?:de\s+edad|cumplidos?)',
        r'(\d{1,3})\s*a[ñn]os?',
    ]:
        m = re.search(p, txt, re.I)
        if m:
            c = int(m.group(1))
            if 0 < c <= 120:
                edad = c
                break

    # Sexo
    m_sx = re.search(r'\b(masculino|femenino|hombre|mujer|var[oó]n|femenina|ni[ñn]o|ni[ñn]a|se[ñn]ora|se[ñn]or)\b', txt, re.I)
    sexo = None
    if m_sx:
        sexo = SEXO_MAP.get(_norm(m_sx.group(1)))

    ps, pd = _pa(txt)
    gl = _first([
        # FIX: "glucosa en ayunas de 95 mg" — agregado "en ayunas"
        r'glucos[ao]\s+(?:en\s+ayunas?\s+(?:de\s+)?|de\s+|capilar\s+)?(\d{2,3})',
        r'glucemia\s+(?:en\s+ayunas?\s+(?:de\s+)?|capilar\s+)?(\d{2,3})',
        r'azucar\s+en\s+sangre\s+(\d{2,3})',
        r'azucar\s+\w+\s+\w+,?\s*(\d{2,3})',
    ], txt)
    tc = _temp(txt)
    fc = _first([
        # FIX: "cardíaca" → txt normalizado lo convierte a "cardiaca" automáticamente
        r'frecuencia\s+cardiaca\s+(?:de\s+)?(\d{2,3})\s*(?:latidos?(?:\s+por\s+minuto)?|lpm|por\s+minuto)',
        r'pulso\s+(\d{2,3})',
        r'FC[:\s]+?(\d{2,3})',
        r'ritmo\s+cardiaco\s+(\d{2,3})',
    ], txt)
    pk = _first([
        # FIX: "peso de 78 kg" — agregado "(?:de\s+)?"
        r'peso\s+(?:de\s+)?(?:corporal[:\s]+)?(\d{2,3}(?:[.,]\d)?)\s*(?:kilogramos?|kg)',
        r'pesa\s+(\d{2,3}(?:[.,]\d)?)\s*(?:kg|kilos?)',
        r'(\d{2,3}(?:[.,]\d)?)\s*kilos?\b',
    ], txt)

    # FIX: talla en metros "1.72 m" → convertir a cm multiplicando ×100
    tl = None
    m_metros = re.search(r'talla\s+(?:de\s+)?(\d)[.,](\d{2})\s*m(?:etros?)?\b', txt, re.I)
    if m_metros:
        tl = str(round(float(f"{m_metros.group(1)}.{m_metros.group(2)}") * 100, 1))
    if tl is None:
        tl = _first([
            r'talla\s+(?:de\s+)?(\d{2,3}(?:[.,]\d)?)\s*(?:centimetros?|cm)',
            r'estatura\s+(?:de\s+)?(\d{2,3}(?:[.,]\d)?)\s*(?:cm|centimetros?)',
            r'mide\s+(\d{2,3}(?:[.,]\d)?)\s*cm',
            r'(\d{2,3}(?:[.,]\d)?)\s*(?:cm|centimetros?)\b',
        ], txt)
    dur = _first([
        r'(\d{1,3})\s*d[ií]as?\s+de\s+evoluci[oó]n',
        r'desde\s+hace\s+(\d{1,3})\s*d[ií]as?',
        r'lleva\s+(\d{1,3})\s*d[ií]as?',
        r'(\d{1,3})\s*d[ií]as?',
    ], txt)

    # Categoría — usar txt normalizado (sin tildes) para comparar con keywords también normalizados
    cat_scores = {c: sum(_norm(kw) in txt for kw in kws) for c, kws in CATEGORIA_KEYWORDS.items()}
    categoria = max(cat_scores, key=cat_scores.get) if max(cat_scores.values()) > 0 else None

    return {
        'edad':                    edad,
        'sexo':                    sexo,
        'peso_kg':                 round(float(pk), 1) if pk else None,
        'talla_cm':                round(float(tl), 1) if tl else None,
        'presion_sistolica':       ps,
        'presion_diastolica':      pd,
        'glucosa_mg_dl':           int(float(gl)) if gl else None,
        'temperatura_c':           tc,
        'frecuencia_cardiaca_bpm': int(float(fc)) if fc else None,
        'duracion_sintomas_dias':  int(float(dur)) if dur else None,
        'categoria_sintoma':       categoria,
    }


def extraer_variables_claude(texto: str) -> dict:
    """
    Extrae variables usando Claude API directamente (sin Docker).
    No hay reglas — el LLM entiende cualquier fraseo médico en español.
    """
    import anthropic, json, os
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = (
        "Eres un extractor de variables clínicas. "
        "Devuelve SOLO un JSON sin texto adicional."
    )
    prompt = f"""Del siguiente texto médico en español extrae estas variables.
Usa null si no aparece. Talla en metros → convertir a cm. Sexo: M o F.
categoria_sintoma: Respiratorio | Gastrointestinal | Hipertensión | Diabetes |
Vacunación | Nutrición | Embarazo | Traumatología | Dermatológico | Infeccioso/Vectorial

Texto: "{texto}"

{{
  "edad": null, "sexo": null, "peso_kg": null, "talla_cm": null,
  "presion_sistolica": null, "presion_diastolica": null,
  "glucosa_mg_dl": null, "temperatura_c": null,
  "frecuencia_cardiaca_bpm": null, "duracion_sintomas_dias": null,
  "categoria_sintoma": null
}}"""

    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw), None
    except Exception:
        return {}, None


def extraer_variables_api(texto: str) -> dict:
    """Envía el texto al microservicio Docker en localhost:8000."""
    import urllib.request, urllib.error
    payload = json.dumps({"texto": texto}).encode('utf-8')
    req = urllib.request.Request(
        "http://localhost:8000/consulta-completa",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("extraccion", {}), data.get("anomalia")
    except urllib.error.URLError:
        print(f"\n{ROJO}No se pudo conectar al microservicio (localhost:8000).{RESET}")
        print(f"{AMARILLO}Asegúrate de que Docker esté corriendo: docker-compose up{RESET}")
        return {}, None


def mostrar_variables(variables: dict, anomalia: dict = None):
    """Imprime SOLO los datos valiosos extraídos, en formato claro."""
    ETIQUETAS = {
        'edad':                    ('Edad',               'años'),
        'sexo':                    ('Sexo',               ''),
        'peso_kg':                 ('Peso',               'kg'),
        'talla_cm':                ('Talla',              'cm'),
        'presion_sistolica':       ('Presión sistólica',  'mmHg'),
        'presion_diastolica':      ('Presión diastólica', 'mmHg'),
        'glucosa_mg_dl':           ('Glucosa',            'mg/dL'),
        'temperatura_c':           ('Temperatura',        '°C'),
        'frecuencia_cardiaca_bpm': ('Frec. cardíaca',     'lpm'),
        'duracion_sintomas_dias':  ('Duración síntomas',  'días'),
        'categoria_sintoma':       ('Categoría clínica',  ''),
    }

    print()
    print(f"  {BOLD}{CYAN}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}{CYAN}║     VARIABLES CLÍNICAS EXTRAÍDAS                ║{RESET}")
    print(f"  {BOLD}{CYAN}╚══════════════════════════════════════════════════╝{RESET}")
    print()

    detectados = 0
    for campo, (etiqueta, unidad) in ETIQUETAS.items():
        valor = variables.get(campo)
        if valor is not None:
            txt_val = f"{valor} {unidad}".strip()
            # Alertas visuales en valores críticos
            alerta = ""
            if campo == 'glucosa_mg_dl' and isinstance(valor, (int, float)) and valor > 200:
                alerta = f"  {ROJO}⚠ ALTO{RESET}"
            elif campo == 'temperatura_c' and isinstance(valor, float) and valor > 38.0:
                alerta = f"  {AMARILLO}⚠ FIEBRE{RESET}"
            elif campo == 'presion_sistolica' and isinstance(valor, int) and valor > 140:
                alerta = f"  {AMARILLO}⚠ HIPERTENSIÓN{RESET}"
            print(f"  {VERDE}✓{RESET}  {etiqueta:<25} {BOLD}{txt_val}{RESET}{alerta}")
            detectados += 1
        else:
            print(f"  {GRIS}✗  {etiqueta:<25} No detectado{RESET}")

    print()

    # Resultado de anomalía si viene del microservicio
    if anomalia is not None:
        es_anomalia  = anomalia.get('es_anomalia')
        score        = anomalia.get('score', 0)
        nivel        = anomalia.get('nivel_riesgo', '')
        if es_anomalia:
            color_nivel = ROJO if nivel == 'anomalo' else AMARILLO
            print(f"  {ROJO}🚨 ALERTA EPIDEMIOLÓGICA{RESET}")
            print(f"     Isolation Forest: {color_nivel}{BOLD}{nivel.upper()}{RESET}  (score={score:.4f})")
        else:
            print(f"  {VERDE}✅ Sin alerta epidemiológica{RESET}  (score={score:.4f})")
        print()

    total_campos = len(ETIQUETAS)
    pct = detectados / total_campos * 100
    color_pct = VERDE if pct >= 80 else AMARILLO if pct >= 50 else ROJO
    print(f"  {color_pct}Campos detectados: {detectados}/{total_campos} ({pct:.0f}%){RESET}")
    print()


def guardar_sesion(timestamp: str, texto_completo: str, variables: dict,
                   wav_path: str, anomalia: dict = None):
    """Guarda texto completo + variables en un JSON y mueve el WAV."""
    nombre_base = timestamp.replace(':', '-').replace(' ', '_')
    json_path   = SESIONES_DIR / f"{nombre_base}.json"
    wav_dest    = SESIONES_DIR / f"{nombre_base}.wav"

    # Mover WAV
    import shutil
    shutil.move(wav_path, str(wav_dest))

    datos = {
        "timestamp":       timestamp,
        "transcripcion":   texto_completo,
        "variables":       variables,
        "anomalia":        anomalia,
        "audio":           str(wav_dest),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    print(f"  {VERDE}Sesión guardada:{RESET}")
    print(f"    {GRIS}Audio:    {wav_dest}{RESET}")
    print(f"    {GRIS}Datos:    {json_path}{RESET}")


# ── Programa principal ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper', action='store_true',
                        help='Usar Whisper (offline) en lugar de Google')
    parser.add_argument('--api', action='store_true',
                        help='Enviar al microservicio Docker (necesita docker-compose up)')
    args = parser.parse_args()

    import os as _os
    tiene_key = bool(_os.environ.get("ANTHROPIC_API_KEY"))
    modo_transcripcion = "Whisper (offline)" if args.whisper else "Google Speech (online)"
    if args.api:
        modo_extraccion = "Microservicio Docker"
    elif tiene_key:
        modo_extraccion = "Claude API (LLM real — sin reglas)"
    else:
        modo_extraccion = "Regex local (offline)"

    # ── Cabecera ────────────────────────────────────────────────────────────
    print()
    print(f"  {BOLD}{MAGENTA}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}{MAGENTA}║   EpiDiagnostix-Mayab — Prueba de Voz          ║{RESET}")
    print(f"  {BOLD}{MAGENTA}╚══════════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  Transcripción : {CYAN}{modo_transcripcion}{RESET}")
    print(f"  Extracción    : {CYAN}{modo_extraccion}{RESET}")
    print()

    # ── Paso 1: Grabar ──────────────────────────────────────────────────────
    print(f"  {BOLD}PASO 1 — Grabación{RESET}")
    print(f"  Presiona {BOLD}ENTER{RESET} para empezar a grabar la consulta...")
    input()
    print()
    print(f"  {BOLD}Habla ahora.{RESET} Presiona {BOLD}ENTER{RESET} cuando termines.")
    print()

    audio = grabar_hasta_enter()

    if len(audio) < SAMPLE_RATE:     # menos de 1 segundo
        print(f"  {ROJO}Audio muy corto — intenta de nuevo.{RESET}")
        sys.exit(1)

    duracion_seg = len(audio) / SAMPLE_RATE
    print(f"  {VERDE}Grabación completada: {duracion_seg:.1f} segundos{RESET}")
    print()

    # Guardar WAV temporal
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp_wav   = tempfile.mktemp(suffix='.wav')
    guardar_wav(audio, tmp_wav)

    # ── Paso 2: Transcribir ─────────────────────────────────────────────────
    print(f"  {BOLD}PASO 2 — Transcripción{RESET}")
    print(f"  {GRIS}Procesando audio...{RESET}")

    if args.whisper:
        texto = transcribir_whisper(tmp_wav)
    else:
        texto = transcribir_google(tmp_wav)

    if not texto.strip():
        print(f"  {ROJO}No se pudo transcribir el audio.{RESET}")
        print(f"  Sugerencias:")
        print(f"    • Habla más cerca del micrófono")
        print(f"    • Prueba con --whisper para modo offline")
        print(f"    • Verifica que tienes conexión a internet (modo Google)")
        sys.exit(1)

    print()
    print(f"  {BOLD}Transcripción completa (guardada):{RESET}")
    print(f"  {GRIS}{'─'*52}{RESET}")
    # Mostrar en líneas de max 70 chars
    palabras = texto.split()
    linea = "  "
    for p in palabras:
        if len(linea) + len(p) > 72:
            print(f"{GRIS}{linea}{RESET}")
            linea = "  " + p + " "
        else:
            linea += p + " "
    if linea.strip():
        print(f"{GRIS}{linea}{RESET}")
    print(f"  {GRIS}{'─'*52}{RESET}")
    print()

    # ── Paso 3: Extraer variables ───────────────────────────────────────────
    print(f"  {BOLD}PASO 3 — Extracción de variables clínicas{RESET}")
    print(f"  {GRIS}Analizando...{RESET}")
    print()

    anomalia = None
    if args.api:
        variables, anomalia = extraer_variables_api(texto)
    elif tiene_key:
        variables, _ = extraer_variables_claude(texto)
    else:
        variables = extraer_variables_local(texto)

    mostrar_variables(variables, anomalia)

    # ── Paso 4: Guardar ─────────────────────────────────────────────────────
    resp = input(f"  ¿Guardar sesión completa? {BOLD}[S/n]{RESET}: ").strip().lower()
    if resp in ('', 's', 'si', 'sí', 'y', 'yes'):
        guardar_sesion(timestamp, texto, variables, tmp_wav, anomalia)
    else:
        os.remove(tmp_wav)
        print(f"  {GRIS}Sesión descartada.{RESET}")

    print()
    print(f"  {BOLD}{VERDE}¡Listo!{RESET}")
    print()


if __name__ == '__main__':
    main()
