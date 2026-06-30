"""
Post-procesador de desambiguación para el NER BiLSTM.
Dos reglas separadas, derivadas de los patrones RE_DUR de extractor_adapter.py.
"""
import sys, io, re, json, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ── Vocabulario / modelo ─────────────────────────────────────────────────────
with open("models/tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")

# ── Tokenizador ───────────────────────────────────────────────────────────────
_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
def tokenizar(texto):
    return [m.group() for m in _RE.finditer(texto.lower())]

def inferir_raw(texto):
    """Retorna (tokens, labels) SIN post-procesamiento."""
    tokens = tokenizar(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens[:40]]
    x = np.array([ids + [0]*(40-len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    return tokens, [ID2LABEL[i] for i in pred]

# ── Tokens de contexto de duración (derivados de RE_DUR en extractor_adapter) ─
# Fuente: d[ií]as? → "dias","dia"
#         evoluci[oó]n → "evolucion"
#         cuadro, lleva, desde, hace → literales
DUR_CONTEXT = {"dias", "dia", "evolucion", "cuadro", "lleva", "desde", "hace"}
VENTANA_CONTEXTO = 4   # tokens antes y después del candidato

# Tokens de género para validar que el span SEXO tiene palabra real de género
SEXO_TOKENS = {
    "mujer","hombre","varon","masculino","femenino","femenina","senora","senor",
    "nina","nino","dama","caballero","muchacha","muchacho"
}

def aplicar_reglas(tokens, labels):
    """
    Aplica las dos reglas de desambiguación en orden:
      Regla 2 primero (fix SEXO con número atrapado)
      Regla 1 después  (fix duplicado B-EDAD → B-DURACION)
    El orden importa: Regla 2 puede crear un segundo B-EDAD que Regla 1 luego convierte.
    """
    labels = list(labels)  # copia mutable

    # ── REGLA 2: número atrapado dentro de span SEXO ─────────────────────────
    # Condición: token etiquetado B-SEXO que sea numérico,
    #            seguido de token con I-SEXO que sea "anos" o "an"
    i = 0
    while i < len(labels):
        if i < len(tokens) and labels[i] == "B-SEXO" and tokens[i].isdigit():
            # El token i es un número con etiqueta B-SEXO
            # Verificar que la raíz del span (tokens antes con B-SEXO) tiene un token de género
            hay_genero_antes = any(
                labels[j] in ("B-SEXO", "I-SEXO") and tokens[j] in SEXO_TOKENS
                for j in range(max(0, i-5), i)
            )
            siguiente_es_anos = (
                i + 1 < len(tokens) and
                labels[i+1] == "I-SEXO" and
                tokens[i+1] in ("anos", "an", "a")
            )
            if hay_genero_antes and siguiente_es_anos:
                labels[i]   = "B-EDAD"      # número → EDAD
                labels[i+1] = "O"           # "anos" → contexto, no entidad
        i += 1

    # ── REGLA 1: duplicado de B-EDAD ─────────────────────────────────────────
    # Condición: más de un token con etiqueta B-EDAD en la misma secuencia.
    # Acción: el primero se queda como EDAD; los siguientes se re-etiquetan
    #         como B-DURACION si tienen tokens de contexto DUR cerca.
    edad_indices = [i for i, l in enumerate(labels) if l == "B-EDAD"]
    if len(edad_indices) > 1:
        for idx in edad_indices[1:]:   # saltar el primero (correcto)
            ini = max(0, idx - VENTANA_CONTEXTO)
            fin = min(len(tokens), idx + VENTANA_CONTEXTO + 1)
            contexto = set(tokens[ini:fin]) - {tokens[idx]}
            if contexto & DUR_CONTEXT:   # intersección no vacía
                labels[idx] = "B-DURACION"

    return labels


# ── Extractor de campos desde labels post-procesadas ─────────────────────────
def norm(s): return unicodedata.normalize("NFD",str(s)).encode("ascii","ignore").decode().lower().strip()
def to_int(v):
    try: return int(float(str(v).replace(",",".")))
    except: return None
def to_float(v):
    try: return round(float(str(v).replace(",",".")),1)
    except: return None

SEXO_MAP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M","caballero":"M",
            "mujer":"F","femenino":"F","femenina":"F","nina":"F","senora":"F","dama":"F"}
CAT_MAP  = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
            "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
            "vacunacion":"Vacunacion","nutricion":"Nutricion"}

def extraer_desde_labels(tokens, labels):
    res = {}
    for tok, lbl in zip(tokens, labels):
        if lbl == "O" or not lbl.startswith("B-"): continue
        base = lbl[2:]
        if base == "EDAD" and "edad" not in res:
            v = to_int(tok)
            if v and 0 < v <= 120: res["edad"] = v
        elif base == "SEXO" and "sexo" not in res:
            res["sexo"] = SEXO_MAP.get(norm(tok))
        elif base == "PESO_KG"  and "peso_kg" not in res: res["peso_kg"] = to_float(tok)
        elif base == "TALLA_CM" and "talla_cm" not in res:
            v = to_float(tok); res["talla_cm"] = round(v*100,1) if v and v<3 else v
        elif base == "PRESION_SIS" and "presion_sistolica" not in res: res["presion_sistolica"] = to_int(tok)
        elif base == "PRESION_DIA" and "presion_diastolica" not in res: res["presion_diastolica"] = to_int(tok)
        elif base == "GLUCOSA"    and "glucosa_mg_dl" not in res: res["glucosa_mg_dl"] = to_int(tok)
        elif base == "TEMPERATURA" and "temperatura_c" not in res: res["temperatura_c"] = to_float(tok)
        elif base == "FREC_CARD"  and "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"] = to_int(tok)
        elif base == "DURACION"   and "duracion_sintomas_dias" not in res:
            v = to_int(tok)
            if v is not None: res["duracion_sintomas_dias"] = v
        elif base == "CATEGORIA"  and "categoria_sintoma" not in res: res["categoria_sintoma"] = CAT_MAP.get(norm(tok), tok)
    return res

# ── Función completa de inferencia + post-procesamiento ──────────────────────
def inferir(texto, con_reglas=True):
    tokens, labels_raw = inferir_raw(texto)
    labels_pp = aplicar_reglas(tokens, labels_raw) if con_reglas else labels_raw
    return extraer_desde_labels(tokens, labels_pp), extraer_desde_labels(tokens, labels_raw)

# ── Evaluación ────────────────────────────────────────────────────────────────
TOL = {"edad":1,"peso_kg":1,"talla_cm":2,"presion_sistolica":1,"presion_diastolica":1,
       "glucosa_mg_dl":1,"temperatura_c":0.2,"frecuencia_cardiaca_bpm":1,"duracion_sintomas_dias":1}
def ok(pred, esp, campo):
    if pred is None: return False
    tol = TOL.get(campo)
    if tol: return abs(float(pred) - float(esp)) <= tol
    return norm(str(pred)) == norm(str(esp))

# ── Datos de evaluación ───────────────────────────────────────────────────────
SET2 = [
    ("SOAP clasico",
     "Consulta medica. Paciente masculino de 45 anos de edad. "
     "Peso 82.0 kg, talla 170.5 cm. "
     "Motivo: dolor abdominal, diarrea, deshidratacion, 5 dias de evolucion. "
     "Signos vitales: presion arterial de 125/82 mmHg, glucosa de 98 mg/dl, "
     "temperatura 37.2°C, frecuencia cardiaca de 91 latidos por minuto.",
     {"edad":45,"sexo":"M","peso_kg":82.0,"talla_cm":170.5,"presion_sistolica":125,
      "presion_diastolica":82,"glucosa_mg_dl":98,"temperatura_c":37.2,
      "frecuencia_cardiaca_bpm":91,"duracion_sintomas_dias":5}),
    ("Dictado rapido",
     "Paciente femenina, 67 anos. PA: 160/95, glucemia capilar 310 mg/dL. "
     "T: 36.8 grados. FC: 88 lpm. Peso 63.5 kg, estatura de 155.0 cm. "
     "Sintomas: fatiga y perdida de peso desde hace 3 dias.",
     {"edad":67,"sexo":"F","peso_kg":63.5,"talla_cm":155.0,"presion_sistolica":160,
      "presion_diastolica":95,"glucosa_mg_dl":310,"temperatura_c":36.8,
      "frecuencia_cardiaca_bpm":88,"duracion_sintomas_dias":3}),
    ("Coloquial comunidad",
     "Se atiende a mujer de 32 anos que lleva 7 dias con tos y dificultad para respirar. "
     "Presion 118 sobre 74, azucar en sangre 89, temperatura 38.5, pulso 102. "
     "Pesa 58 kilos, mide 162 cm.",
     {"edad":32,"sexo":"F","peso_kg":58.0,"talla_cm":162.0,"presion_sistolica":118,
      "presion_diastolica":74,"glucosa_mg_dl":89,"temperatura_c":38.5,
      "frecuencia_cardiaca_bpm":102,"duracion_sintomas_dias":7}),
    ("Pediatrico vacunacion",
     "Paciente masculino, edad: 2 anos. Consulta por fiebre leve postvacuna "
     "y control de esquema de vacunacion con 1 dia de evolucion. "
     "Peso corporal: 12.5 kg, talla 82.0 centimetros. "
     "Tension arterial 100/65 mmHg, temperatura de 37.8 grados, "
     "frecuencia cardiaca de 110 lpm, glucosa 88 mg/dl.",
     {"edad":2,"sexo":"M","peso_kg":12.5,"talla_cm":82.0,"presion_sistolica":100,
      "presion_diastolica":65,"glucosa_mg_dl":88,"temperatura_c":37.8,
      "frecuencia_cardiaca_bpm":110,"duracion_sintomas_dias":1}),
    ("Texto libre sin estructura",
     "La senora tiene como 55 anos, es hipertensa, refiere zumbido en los oidos "
     "y vision borrosa desde hace 4 dias. Le tome la presion y salio 170 sobre 100. "
     "Su azucar estaba bien, 95. Temperatura normal 36.5. Pulso 78. "
     "Pesa como 70 kilos y mide 158 centimetros.",
     {"edad":55,"sexo":"F","peso_kg":70.0,"talla_cm":158.0,"presion_sistolica":170,
      "presion_diastolica":100,"glucosa_mg_dl":95,"temperatura_c":36.5,
      "frecuencia_cardiaca_bpm":78,"duracion_sintomas_dias":4}),
]

SET3 = [
    ("Ciego 01","Tension arterial 128/82. Calentura de 37.9 grados. Glucemia 91 mg/dl. FC 76 lpm. Senora de 52 anos. Peso 68 kilos, estatura 160 cm. Desde hace 5 dias.",{"sexo":"F","edad":52,"presion_sistolica":128,"presion_diastolica":82,"temperatura_c":37.9,"glucosa_mg_dl":91,"frecuencia_cardiaca_bpm":76,"peso_kg":68,"talla_cm":160,"duracion_sintomas_dias":5}),
    ("Ciego 02","Pues este paciente varon de 33 anos llego con 3 dias de evolucion. Peso 82 kg, talla 175 cm. Le tome la tension: 118/76 mmHg. Su azucar en sangre estaba 88. Temperatura 37.1. Frecuencia cardiaca 80.",{"sexo":"M","edad":33,"duracion_sintomas_dias":3,"peso_kg":82,"talla_cm":175,"presion_sistolica":118,"presion_diastolica":76,"glucosa_mg_dl":88,"temperatura_c":37.1,"frecuencia_cardiaca_bpm":80}),
    ("Ciego 03","Paciente femenino de 26 anos. Signos: TA 110/70, T 36.5, FC 72, Glx 85 mg/dl. Peso 55.5 kg, talla 1.58 m. Tiempo de evolucion: 7 dias.",{"sexo":"F","edad":26,"presion_sistolica":110,"presion_diastolica":70,"temperatura_c":36.5,"frecuencia_cardiaca_bpm":72,"glucosa_mg_dl":85,"peso_kg":55.5,"talla_cm":158,"duracion_sintomas_dias":7}),
    ("Ciego 04","Se atiende a caballero de 71 anos. Calentura de 38.1 grados. Presion 162/98 mmHg. Pulso 94 latidos. Nivel de glucosa 145. Peso 79 kg, mide 163 cm. Lleva una semana enfermo.",{"sexo":"M","edad":71,"temperatura_c":38.1,"presion_sistolica":162,"presion_diastolica":98,"frecuencia_cardiaca_bpm":94,"glucosa_mg_dl":145,"peso_kg":79,"talla_cm":163,"duracion_sintomas_dias":7}),
    ("Ciego 05","Doctor, le informo sobre el nino de 7 anos. Masa corporal 22 kg, altura 118 cm. Temperatura axilar 39.2 grados. Ritmo cardiaco 115 bpm. PA 90/55. Inicio hace 2 dias.",{"sexo":"M","edad":7,"peso_kg":22,"talla_cm":118,"temperatura_c":39.2,"frecuencia_cardiaca_bpm":115,"presion_sistolica":90,"presion_diastolica":55,"duracion_sintomas_dias":2}),
    ("Ciego 06","Paciente masculino, 54 anos, 80.5 kilogramos, 172 centimetros. Tension: 140/88. Glicemia en ayunas: 118. Temperatura: 36.9. Pulso: 84. Cuadro de 4 dias.",{"sexo":"M","edad":54,"peso_kg":80.5,"talla_cm":172,"presion_sistolica":140,"presion_diastolica":88,"glucosa_mg_dl":118,"temperatura_c":36.9,"frecuencia_cardiaca_bpm":84,"duracion_sintomas_dias":4}),
    ("Ciego 07","Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78.",{"sexo":"F","edad":44,"duracion_sintomas_dias":9,"peso_kg":61,"talla_cm":155,"temperatura_c":37.3,"glucosa_mg_dl":97,"presion_sistolica":125,"presion_diastolica":80,"frecuencia_cardiaca_bpm":78}),
    ("Ciego 08","Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, FC 68, glucosa 92. Cuadro agudo de 1 dia.",{"sexo":"M","edad":18,"presion_sistolica":108,"presion_diastolica":65,"temperatura_c":36.7,"frecuencia_cardiaca_bpm":68,"glucosa_mg_dl":92,"duracion_sintomas_dias":1}),
    ("Ciego 09","Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88.",{"sexo":"F","edad":65,"duracion_sintomas_dias":10,"presion_sistolica":175,"presion_diastolica":105,"temperatura_c":37.4,"frecuencia_cardiaca_bpm":88}),
    ("Ciego 10","Escolar femenino de 11 anos. Talla 140 cm, peso 36 kg. Temperatura de 38.6 grados. Pulso 98. Presion 95/60. Glucemia 88 mg/dl. 3 dias.",{"sexo":"F","edad":11,"talla_cm":140,"peso_kg":36,"temperatura_c":38.6,"frecuencia_cardiaca_bpm":98,"presion_sistolica":95,"presion_diastolica":60,"glucosa_mg_dl":88,"duracion_sintomas_dias":3}),
    ("Ciego 11","Hombre de 47 anos. Viene por cuadro de 8 dias. Peso actual: 91 kg. Estatura: 178 cm. Signos: tension 138/90, azucar 126 mg/dl, T 36.6C, latidos 82 por minuto.",{"sexo":"M","edad":47,"duracion_sintomas_dias":8,"peso_kg":91,"talla_cm":178,"presion_sistolica":138,"presion_diastolica":90,"glucosa_mg_dl":126,"temperatura_c":36.6,"frecuencia_cardiaca_bpm":82}),
    ("Ciego 12","Me llego este paciente femenino de 29 anos, con fiebre de 39.5 y llevaba 4 dias asi. La presion la tenia en 105/68. El pulso en 102.",{"sexo":"F","edad":29,"temperatura_c":39.5,"duracion_sintomas_dias":4,"presion_sistolica":105,"presion_diastolica":68,"frecuencia_cardiaca_bpm":102}),
    ("Ciego 13","Datos del paciente: edad 58 anos, sexo masculino. Medidas: 74 kg / 166 cm. Signos vitales: tension 148/92 mmHg, glucemia 210 mg/dl, temperatura 37.0 grados, FC 88 lpm. Evolucion: 6 dias.",{"edad":58,"sexo":"M","peso_kg":74,"talla_cm":166,"presion_sistolica":148,"presion_diastolica":92,"glucosa_mg_dl":210,"temperatura_c":37.0,"frecuencia_cardiaca_bpm":88,"duracion_sintomas_dias":6}),
    ("Ciego 14","Adulto mayor de sexo femenino, 78 anos. Refiere inicio hace 2 dias. Peso 52 kg. Talla 149 cm. La calentura: 37.8. Azucar en ayunas: 135. Tension: 168/102. Latidos: 92.",{"sexo":"F","edad":78,"duracion_sintomas_dias":2,"peso_kg":52,"talla_cm":149,"temperatura_c":37.8,"glucosa_mg_dl":135,"presion_sistolica":168,"presion_diastolica":102,"frecuencia_cardiaca_bpm":92}),
    ("Ciego 15","Paciente de 40 anos, sexo masculino. Sin datos de peso y talla. Tension arterial: 122/80. Temperatura: 38.0C. Glucosa: 99. Frecuencia cardiaca: 90. Evolucion de 3 dias.",{"edad":40,"sexo":"M","presion_sistolica":122,"presion_diastolica":80,"temperatura_c":38.0,"glucosa_mg_dl":99,"frecuencia_cardiaca_bpm":90,"duracion_sintomas_dias":3}),
    ("Ciego 16","Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. Peso 67.5 kg. Estatura 1.74 m.",{"sexo":"M","edad":25,"duracion_sintomas_dias":5,"presion_sistolica":115,"presion_diastolica":72,"temperatura_c":36.4,"frecuencia_cardiaca_bpm":70,"glucosa_mg_dl":90,"peso_kg":67.5,"talla_cm":174}),
    ("Ciego 17","Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. Le registre tension de 120/78 mmHg. Nivel de glucosa 88. Temperatura axilar 37.2. Pulso 76 por minuto.",{"sexo":"F","edad":36,"duracion_sintomas_dias":11,"peso_kg":58,"talla_cm":162,"presion_sistolica":120,"presion_diastolica":78,"glucosa_mg_dl":88,"temperatura_c":37.2,"frecuencia_cardiaca_bpm":76}),
    ("Ciego 18","Nino de 5 anos, masculino. Fiebre 38.4C de 3 dias de evolucion. Peso 18 kg, mide 108 cm. Tension 88/58. FC 108. Glucosa 85.",{"sexo":"M","edad":5,"temperatura_c":38.4,"duracion_sintomas_dias":3,"peso_kg":18,"talla_cm":108,"presion_sistolica":88,"presion_diastolica":58,"frecuencia_cardiaca_bpm":108,"glucosa_mg_dl":85}),
    ("Ciego 19","Consulto paciente femenino de 62 anos con 14 dias de cuadro. Mide 1.53 m, pesa 71 kilos. Tension 158/96. Calentura 37.6. Azucar en sangre 142 mg/dl. Pulso 86 lpm.",{"sexo":"F","edad":62,"duracion_sintomas_dias":14,"talla_cm":153,"peso_kg":71,"presion_sistolica":158,"presion_diastolica":96,"temperatura_c":37.6,"glucosa_mg_dl":142,"frecuencia_cardiaca_bpm":86}),
    ("Ciego 20","Masculino, 15 anos. Talla 165 cm, peso 55 kg. Temperatura de 39.0 grados. Pulso 105. Tension arterial 100/65. Glucosa 90. Inicio hace 1 dia.",{"sexo":"M","edad":15,"talla_cm":165,"peso_kg":55,"temperatura_c":39.0,"frecuencia_cardiaca_bpm":105,"presion_sistolica":100,"presion_diastolica":65,"glucosa_mg_dl":90,"duracion_sintomas_dias":1}),
]

# ── Evaluación comparativa ────────────────────────────────────────────────────
def evaluar(nombre_set, casos, mostrar_cambios=True):
    ok_sin = ok_con = total = 0
    cambios = []
    for nombre, texto, esp in casos:
        res_con, res_sin = inferir(texto, con_reglas=True)
        for campo, val_esp in esp.items():
            pred_sin = res_sin.get(campo)
            pred_con = res_con.get(campo)
            antes = ok(pred_sin, val_esp, campo)
            despues = ok(pred_con, val_esp, campo)
            ok_sin += antes
            ok_con += despues
            total += 1
            if antes != despues:
                cambios.append({
                    "caso": nombre, "campo": campo,
                    "esperado": val_esp,
                    "sin_reglas": pred_sin, "con_reglas": pred_con,
                    "tipo": "MEJORA" if despues and not antes else "REGRESION"
                })

    print(f"\n{'='*65}")
    print(f"  {nombre_set}")
    print(f"  Sin reglas: {ok_sin}/{total} ({ok_sin/total*100:.1f}%)")
    print(f"  Con reglas: {ok_con}/{total} ({ok_con/total*100:.1f}%)")
    delta = ok_con - ok_sin
    print(f"  Cambio:     {'+' if delta>=0 else ''}{delta} campos")

    if cambios and mostrar_cambios:
        print(f"\n  Cambios campo a campo:")
        for c in cambios:
            print(f"    [{c['tipo']}] {c['caso']} | {c['campo']}: "
                  f"esp={c['esperado']} | sin={c['sin_reglas']} -> con={c['con_reglas']}")
    return ok_sin, ok_con, total, cambios

# Evaluación completa
ok_s2_sin, ok_s2_con, t2, cambios2 = evaluar("SET 2 — 5 textos de referencia (50 campos)", SET2)
ok_s3_sin, ok_s3_con, t3, cambios3 = evaluar("SET 3 — 20 frases ciegas", SET3)

# ── Confirmaciones específicas del usuario ────────────────────────────────────
print(f"\n{'='*65}")
print("  CONFIRMACIONES ESPECIFICAS")
print(f"{'='*65}")

# 1. Casos 02 y 17 arreglados por Regla 1
for nombre_r, campo_r in [("Ciego 02","duracion_sintomas_dias"), ("Ciego 17","duracion_sintomas_dias")]:
    texto = next(t for n,t,_ in SET3 if n==nombre_r)
    esp   = next(e for n,_,e in SET3 if n==nombre_r)[campo_r]
    tokens, labels_raw = inferir_raw(texto)
    # Solo Regla 1 (sin Regla 2)
    labels_solo_r1 = list(labels_raw)
    edad_idx = [i for i,l in enumerate(labels_solo_r1) if l=="B-EDAD"]
    if len(edad_idx) > 1:
        for idx in edad_idx[1:]:
            ini=max(0,idx-VENTANA_CONTEXTO); fin=min(len(tokens),idx+VENTANA_CONTEXTO+1)
            ctx=set(tokens[ini:fin])-{tokens[idx]}
            if ctx & DUR_CONTEXT: labels_solo_r1[idx]="B-DURACION"
    res_r1 = extraer_desde_labels(tokens, labels_solo_r1)
    corregido = ok(res_r1.get(campo_r), esp, campo_r)
    print(f"\n  1. {nombre_r} corregido por Regla 1 SOLA: {'SI' if corregido else 'NO'}")
    print(f"     duracion esperada={esp}, con solo R1={res_r1.get(campo_r)}")

# 2. Caso 07 — Regla 1 sola vs Regla 2 sola vs ambas
texto_07 = next(t for n,t,_ in SET3 if n=="Ciego 07")
esp_07   = next(e for n,_,e in SET3 if n=="Ciego 07")
tokens_07, labels_raw_07 = inferir_raw(texto_07)

# Solo Regla 1
labels_r1 = list(labels_raw_07)
edad_idx_07 = [i for i,l in enumerate(labels_r1) if l=="B-EDAD"]
if len(edad_idx_07) > 1:
    for idx in edad_idx_07[1:]:
        ini=max(0,idx-VENTANA_CONTEXTO); fin=min(len(tokens_07),idx+VENTANA_CONTEXTO+1)
        if set(tokens_07[ini:fin])-{tokens_07[idx]} & DUR_CONTEXT:
            labels_r1[idx]="B-DURACION"
res_solo_r1 = extraer_desde_labels(tokens_07, labels_r1)

# Solo Regla 2
labels_r2 = list(labels_raw_07)
for i, (t,l) in enumerate(zip(tokens_07, labels_r2)):
    if l=="B-SEXO" and t.isdigit():
        hay_genero = any(labels_r2[j] in ("B-SEXO","I-SEXO") and tokens_07[j] in SEXO_TOKENS for j in range(max(0,i-5),i))
        sig_anos = i+1<len(tokens_07) and labels_r2[i+1]=="I-SEXO" and tokens_07[i+1] in ("anos","an","a")
        if hay_genero and sig_anos:
            labels_r2[i]="B-EDAD"; labels_r2[i+1]="O"
res_solo_r2 = extraer_desde_labels(tokens_07, labels_r2)

# Ambas reglas
labels_ambas = aplicar_reglas(tokens_07, labels_raw_07)
res_ambas = extraer_desde_labels(tokens_07, labels_ambas)

print(f"\n  2. Ciego 07 — edad esp=44, duracion esp=9:")
print(f"     Solo  Regla 1: edad={res_solo_r1.get('edad')} dur={res_solo_r1.get('duracion_sintomas_dias')} "
      f"-> {'CORREGIDO' if ok(res_solo_r1.get('duracion_sintomas_dias'),9,'duracion_sintomas_dias') else 'NO corregido'}")
print(f"     Solo  Regla 2: edad={res_solo_r2.get('edad')} dur={res_solo_r2.get('duracion_sintomas_dias')} "
      f"-> {'CORREGIDO' if ok(res_solo_r2.get('duracion_sintomas_dias'),9,'duracion_sintomas_dias') and ok(res_solo_r2.get('edad'),44,'edad') else 'PARCIAL o NO corregido'}")
print(f"     Ambas reglas:  edad={res_ambas.get('edad')} dur={res_ambas.get('duracion_sintomas_dias')} "
      f"-> {'CORREGIDO' if ok(res_ambas.get('edad'),44,'edad') and ok(res_ambas.get('duracion_sintomas_dias'),9,'duracion_sintomas_dias') else 'NO corregido'}")

# 3. Resumen regresiones
regresiones = [c for c in cambios2+cambios3 if c["tipo"]=="REGRESION"]
print(f"\n  3. Regresiones (casos que antes pasaban y ahora fallan): {len(regresiones)}")
for r in regresiones:
    print(f"     [{r['caso']}] {r['campo']}: esp={r['esperado']} antes={r['sin_reglas']} despues={r['con_reglas']}")
if not regresiones:
    print("     Ninguna — las reglas no rompen nada que antes estuviera bien.")
