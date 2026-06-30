"""
Genera el golden set de paridad para las reglas de post-procesamiento BIO.

Por cada caso (SET2 + SET3):
  - tokens      : lista de tokens (salida del tokenizador)
  - labels_raw  : etiquetas crudas del modelo (entrada a las reglas)
  - labels_pp   : etiquetas tras R1 + R2  (lo que applyRules() debe producir en Dart)
  - fields      : campos extraídos tras R1+R2+R3  (lo que extractFields() debe producir en Dart)
                  Solo se guardan campos con valor no-None.

Salida: models/postproc_golden_set.json
"""
import sys, io, re, json, unicodedata, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ── Modelo y vocabulario ───────────────────────────────────────────────────────
with open("models/tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")

_RE = re.compile(r"\d+[.,]\d+|\d+|[a-z]+")
def tokenizar(t): return [m.group() for m in _RE.finditer(t.lower())][:40]
def norm(s): return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()

def raw_labels(texto):
    tokens = tokenizar(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
    x = np.array([ids + [0] * (40 - len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    return tokens, [ID2LABEL[i] for i in pred]

# ── Constantes (idénticas a postprocess_rules_v2.py) ──────────────────────────
SEXO_TOKENS = {"mujer","hombre","varon","masculino","femenino","femenina",
               "senora","senor","nina","nino","dama","caballero","muchacha","muchacho"}
DUR_CONTEXT = {"dias","dia","evolucion","cuadro","lleva","desde","hace"}
SEXO_MAP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M","caballero":"M",
            "mujer":"F","femenino":"F","femenina":"F","nina":"F","senora":"F","dama":"F"}
CAT_MAP  = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
            "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
            "vacunacion":"Vacunacion","nutricion":"Nutricion"}

# ── Conversores (idénticos a postprocess_rules_v2.py) ─────────────────────────
def to_int(v):
    try: return int(float(str(v).replace(",", ".")))
    except: return None

def to_float(v):
    try: return round(float(str(v).replace(",", ".")), 1)
    except: return None

# ── R1 + R2: mutación de etiquetas ────────────────────────────────────────────
def aplicar_reglas_1_2(tokens, labels):
    labels = list(labels)
    # R2: número atrapado en span SEXO → corregir a B-EDAD
    for i in range(len(labels)):
        if i < len(tokens) and labels[i] == "B-SEXO" and tokens[i].isdigit():
            hay_genero = any(labels[j] in ("B-SEXO", "I-SEXO") and tokens[j] in SEXO_TOKENS
                             for j in range(max(0, i - 5), i))
            sig_anos = (i + 1 < len(tokens) and labels[i + 1] == "I-SEXO"
                        and tokens[i + 1] in ("anos", "an", "a"))
            if hay_genero and sig_anos:
                labels[i] = "B-EDAD"; labels[i + 1] = "O"
    # R1: segundo B-EDAD en contexto de duración → B-DURACION
    edad_idx = [i for i, l in enumerate(labels) if l == "B-EDAD"]
    if len(edad_idx) > 1:
        for idx in edad_idx[1:]:
            ini = max(0, idx - 4); fin = min(len(tokens), idx + 5)
            if set(tokens[ini:fin]) - {tokens[idx]} & DUR_CONTEXT:
                labels[idx] = "B-DURACION"
    return labels

# ── Extractor con R3 integrada ─────────────────────────────────────────────────
def extraer(tokens, labels):
    res = {}
    # R3: SEXO — primer token SEXO (B o I) con valor válido en SEXO_MAP
    for tok, lbl in zip(tokens, labels):
        if "SEXO" in lbl:
            v = SEXO_MAP.get(norm(tok))
            if v is not None:
                res["sexo"] = v
                break
    # Campos restantes (solo tokens B-)
    for tok, lbl in zip(tokens, labels):
        if lbl == "O" or not lbl.startswith("B-"): continue
        base = lbl[2:]
        if base == "EDAD" and "edad" not in res:
            v = to_int(tok)
            if v and 0 < v <= 120: res["edad"] = v
        elif base == "PESO_KG" and "peso_kg" not in res:
            v = to_float(tok)
            if v is not None: res["peso_kg"] = v
        elif base == "TALLA_CM" and "talla_cm" not in res:
            v = to_float(tok)
            if v is not None: res["talla_cm"] = round(v * 100, 1) if v < 3 else v
        elif base == "PRESION_SIS" and "presion_sistolica" not in res:
            v = to_int(tok)
            if v is not None: res["presion_sistolica"] = v
        elif base == "PRESION_DIA" and "presion_diastolica" not in res:
            v = to_int(tok)
            if v is not None: res["presion_diastolica"] = v
        elif base == "GLUCOSA" and "glucosa_mg_dl" not in res:
            v = to_int(tok)
            if v is not None: res["glucosa_mg_dl"] = v
        elif base == "TEMPERATURA" and "temperatura_c" not in res:
            v = to_float(tok)
            if v is not None: res["temperatura_c"] = v
        elif base == "FREC_CARD" and "frecuencia_cardiaca_bpm" not in res:
            v = to_int(tok)
            if v is not None: res["frecuencia_cardiaca_bpm"] = v
        elif base == "DURACION" and "duracion_sintomas_dias" not in res:
            v = to_int(tok)
            if v is not None: res["duracion_sintomas_dias"] = v
        elif base == "CATEGORIA" and "categoria_sintoma" not in res:
            v = CAT_MAP.get(norm(tok))
            if v is not None: res["categoria_sintoma"] = v
    return res

# ── Casos (SET2 + SET3) ────────────────────────────────────────────────────────
CASOS = [
    ("SET2/SOAP",
     "Consulta medica. Paciente masculino de 45 anos de edad. Peso 82.0 kg, talla 170.5 cm. "
     "Motivo: dolor abdominal, diarrea, deshidratacion, 5 dias de evolucion. Signos vitales: "
     "presion arterial de 125/82 mmHg, glucosa de 98 mg/dl, temperatura 37.2°C, "
     "frecuencia cardiaca de 91 latidos por minuto."),
    ("SET2/Dictado",
     "Paciente femenina, 67 anos. PA: 160/95, glucemia capilar 310 mg/dL. T: 36.8 grados. "
     "FC: 88 lpm. Peso 63.5 kg, estatura de 155.0 cm. "
     "Sintomas: fatiga y perdida de peso desde hace 3 dias."),
    ("SET2/Coloquial",
     "Se atiende a mujer de 32 anos que lleva 7 dias con tos y dificultad para respirar. "
     "Presion 118 sobre 74, azucar en sangre 89, temperatura 38.5, pulso 102. "
     "Pesa 58 kilos, mide 162 cm."),
    ("SET2/Pediatrico",
     "Paciente masculino, edad: 2 anos. Consulta por fiebre leve postvacuna y control de "
     "esquema de vacunacion con 1 dia de evolucion. Peso corporal: 12.5 kg, talla 82.0 "
     "centimetros. Tension arterial 100/65 mmHg, temperatura de 37.8 grados, frecuencia "
     "cardiaca de 110 lpm, glucosa 88 mg/dl."),
    ("SET2/TextoLibre",
     "La senora tiene como 55 anos, es hipertensa, refiere zumbido en los oidos y vision "
     "borrosa desde hace 4 dias. Le tome la presion y salio 170 sobre 100. Su azucar estaba "
     "bien, 95. Temperatura normal 36.5. Pulso 78. Pesa como 70 kilos y mide 158 centimetros."),
    ("C01","Tension arterial 128/82. Calentura de 37.9 grados. Glucemia 91 mg/dl. FC 76 lpm. "
           "Senora de 52 anos. Peso 68 kilos, estatura 160 cm. Desde hace 5 dias."),
    ("C02","Pues este paciente varon de 33 anos llego con 3 dias de evolucion. Peso 82 kg, "
           "talla 175 cm. Le tome la tension: 118/76 mmHg. Su azucar en sangre estaba 88. "
           "Temperatura 37.1. Frecuencia cardiaca 80."),
    ("C03","Paciente femenino de 26 anos. Signos: TA 110/70, T 36.5, FC 72, Glx 85 mg/dl. "
           "Peso 55.5 kg, talla 1.58 m. Tiempo de evolucion: 7 dias."),
    ("C04","Se atiende a caballero de 71 anos. Calentura de 38.1 grados. Presion 162/98 mmHg. "
           "Pulso 94 latidos. Nivel de glucosa 145. Peso 79 kg, mide 163 cm. "
           "Lleva una semana enfermo."),
    ("C05","Doctor, le informo sobre el nino de 7 anos. Masa corporal 22 kg, altura 118 cm. "
           "Temperatura axilar 39.2 grados. Ritmo cardiaco 115 bpm. PA 90/55. Inicio hace 2 dias."),
    ("C06","Paciente masculino, 54 anos, 80.5 kilogramos, 172 centimetros. Tension: 140/88. "
           "Glicemia en ayunas: 118. Temperatura: 36.9. Pulso: 84. Cuadro de 4 dias."),
    ("C07","Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. "
           "Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78."),
    ("C08","Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, FC 68, glucosa 92. "
           "Cuadro agudo de 1 dia."),
    ("C09","Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. "
           "La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88."),
    ("C10","Escolar femenino de 11 anos. Talla 140 cm, peso 36 kg. Temperatura de 38.6 grados. "
           "Pulso 98. Presion 95/60. Glucemia 88 mg/dl. 3 dias."),
    ("C11","Hombre de 47 anos. Viene por cuadro de 8 dias. Peso actual: 91 kg. Estatura: 178 cm. "
           "Signos: tension 138/90, azucar 126 mg/dl, T 36.6C, latidos 82 por minuto."),
    ("C12","Me llego este paciente femenino de 29 anos, con fiebre de 39.5 y llevaba 4 dias asi. "
           "La presion la tenia en 105/68. El pulso en 102."),
    ("C13","Datos del paciente: edad 58 anos, sexo masculino. Medidas: 74 kg / 166 cm. "
           "Signos vitales: tension 148/92 mmHg, glucemia 210 mg/dl, temperatura 37.0 grados, "
           "FC 88 lpm. Evolucion: 6 dias."),
    ("C14","Adulto mayor de sexo femenino, 78 anos. Refiere inicio hace 2 dias. Peso 52 kg. "
           "Talla 149 cm. La calentura: 37.8. Azucar en ayunas: 135. Tension: 168/102. Latidos: 92."),
    ("C15","Paciente de 40 anos, sexo masculino. Sin datos de peso y talla. "
           "Tension arterial: 122/80. Temperatura: 38.0C. Glucosa: 99. "
           "Frecuencia cardiaca: 90. Evolucion de 3 dias."),
    ("C16","Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. "
           "TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. "
           "Peso 67.5 kg. Estatura 1.74 m."),
    ("C17","Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. "
           "Le registre tension de 120/78 mmHg. Nivel de glucosa 88. Temperatura axilar 37.2. "
           "Pulso 76 por minuto."),
    ("C18","Nino de 5 anos, masculino. Fiebre 38.4C de 3 dias de evolucion. Peso 18 kg, "
           "mide 108 cm. Tension 88/58. FC 108. Glucosa 85."),
    ("C19","Consulto paciente femenino de 62 anos con 14 dias de cuadro. Mide 1.53 m, pesa 71 "
           "kilos. Tension 158/96. Calentura 37.6. Azucar en sangre 142 mg/dl. Pulso 86 lpm."),
    ("C20","Masculino, 15 anos. Talla 165 cm, peso 55 kg. Temperatura de 39.0 grados. "
           "Pulso 105. Tension arterial 100/65. Glucosa 90. Inicio hace 1 dia."),
]

# ── Generar golden set ─────────────────────────────────────────────────────────
print(f"Generando golden set para {len(CASOS)} casos...")
print("-" * 60)

golden_cases = []
for nombre, texto in CASOS:
    tokens, labels_raw = raw_labels(texto)
    labels_pp = aplicar_reglas_1_2(tokens, labels_raw)
    fields     = extraer(tokens, labels_pp)

    # Registrar cambios de etiqueta para diagnóstico
    cambios = [(i, tokens[i], labels_raw[i], labels_pp[i])
               for i in range(len(tokens)) if labels_raw[i] != labels_pp[i]]
    if cambios:
        print(f"  [{nombre}] cambios de etiqueta:")
        for idx, tok, raw, pp in cambios:
            print(f"    [{idx}] '{tok}': {raw} → {pp}")

    golden_cases.append({
        "id":         nombre,
        "text":       texto,
        "tokens":     tokens,
        "labels_raw": labels_raw,
        "labels_pp":  labels_pp,
        "fields":     fields,
    })

out = {
    "generated":   datetime.date.today().isoformat(),
    "model":       "ner_tflite_v2.keras",
    "n_cases":     len(golden_cases),
    "description": (
        "Para cada caso: tokens (salida tokenizador), labels_raw (modelo crudo), "
        "labels_pp (tras R1+R2), fields (extracción con R3, solo valores no-None). "
        "Dart debe reproducir labels_pp con applyRules() y fields con extractFields()."
    ),
    "cases":       golden_cases,
}

out_path = "models/postproc_golden_set.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\nGuardado: {out_path}")
print(f"Total casos: {len(golden_cases)}")

# Verificación rápida: cuántos casos tienen cambios de etiqueta
con_cambios = sum(1 for c in golden_cases if c["labels_raw"] != c["labels_pp"])
sin_cambios = len(golden_cases) - con_cambios
print(f"Casos con cambios de etiqueta (R1/R2 activos): {con_cambios}")
print(f"Casos sin cambios (reglas no aplican):          {sin_cambios}")
