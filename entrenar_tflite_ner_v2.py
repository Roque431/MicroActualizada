"""
FASE C — Reentrenamiento con dataset v2 + evaluacion en 3 sets:
  1. Test sintetico v2
  2. 55 casos de test_extractor.py (referencia historica)
  3. 20 frases ciegas (metrica de verdad: no derivadas de ninguna plantilla)

Misma arquitectura BiLSTM que valido en presupuesto de tamano.
"""
import os, json, random, warnings, re, unicodedata
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR  = Path("models")
MODEL_OUT = DATA_DIR / "ner_tflite_v2.keras"
META_OUT  = DATA_DIR / "ner_tflite_v2_meta.json"
MAX_LEN   = 40
BATCH_SIZE= 64
EPOCHS    = 35
LR        = 0.001
EMBED_DIM = 64
LSTM1     = 64
LSTM2     = 32

print(f"TensorFlow {tf.__version__}")

with open(DATA_DIR/"tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open(DATA_DIR/"tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
VOCAB_SIZE = len(VOCAB)
N_LABELS   = len(LABEL2ID)
PAD_ID     = VOCAB["<PAD>"]
O_ID       = LABEL2ID["O"]
print(f"Vocab: {VOCAB_SIZE}  Etiquetas: {N_LABELS}")

def cargar_y_padear(path):
    with open(path, encoding="utf-8") as f: data = json.load(f)
    X, y = [], []
    for ej in data:
        ids    = ej["token_ids"][:MAX_LEN]
        labels = ej["label_ids"][:MAX_LEN]
        n = MAX_LEN - len(ids)
        X.append(ids    + [PAD_ID]*n)
        y.append(labels + [O_ID]  *n)
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

print("\nCargando datasets v2...")
X_train, y_train = cargar_y_padear(DATA_DIR/"tflite_train_v2.json")
X_val,   y_val   = cargar_y_padear(DATA_DIR/"tflite_val_v2.json")
X_test,  y_test  = cargar_y_padear(DATA_DIR/"tflite_test_v2.json")
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# Class weights
label_counts = np.bincount(y_train.flatten(), minlength=N_LABELS)
total_tokens = label_counts.sum()
class_weight = {}
for lid in range(N_LABELS):
    if lid == O_ID:
        class_weight[lid] = 0.3
    else:
        freq = label_counts[lid] / total_tokens
        class_weight[lid] = min(1.0 / (freq * N_LABELS), 5.0)

# Modelo (identico al v1 — misma arquitectura)
def construir_modelo():
    inp = keras.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, name="embedding")(inp)
    x = keras.layers.Bidirectional(keras.layers.LSTM(LSTM1, return_sequences=True, dropout=0.2), name="bilstm_1")(x)
    x = keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(LSTM2, return_sequences=True, dropout=0.1), name="bilstm_2")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)
    out = keras.layers.TimeDistributed(keras.layers.Dense(N_LABELS, activation="softmax"), name="output")(x)
    return keras.Model(inputs=inp, outputs=out, name="ner_tflite_v2")

modelo = construir_modelo()
params = modelo.count_params()
print(f"\nParametros: {params:,}  ({params*4/1024**2:.2f} MB fp32)")

modelo.compile(optimizer=keras.optimizers.Adam(LR),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
]

print(f"\nEntrenando v2: {EPOCHS} epocas, batch={BATCH_SIZE}")
historia = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight, callbacks=callbacks, verbose=1,
)

# ── F1 por entidad ────────────────────────────────────────────────────────────
def f1_por_entidad(X, y_true, modelo, id2label):
    y_pred = modelo.predict(X, verbose=0).argmax(axis=-1)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for sp, st in zip(y_pred, y_true):
        for p, t in zip(sp, st):
            pl = id2label[p]; tl = id2label[t]
            if tl == "O" and pl == "O": continue
            if tl != "O":
                if pl == tl: tp[tl] += 1
                else:
                    fn[tl] += 1
                    if pl != "O": fp[pl] += 1
            else:
                fp[pl] += 1
    r = {}
    for l in set(list(tp)+list(fp)+list(fn)):
        p = tp[l]/(tp[l]+fp[l]+1e-9); rec = tp[l]/(tp[l]+fn[l]+1e-9)
        r[l] = {"p":round(p,4),"r":round(rec,4),"f":round(2*p*rec/(p+rec+1e-9),4)}
    return r

print("\nEvaluando en TEST SET sintetico v2...")
metricas_test = f1_por_entidad(X_test, y_test, modelo, ID2LABEL)

CAMPOS = ["EDAD","SEXO","PESO_KG","TALLA_CM","PRESION_SIS","PRESION_DIA",
          "GLUCOSA","TEMPERATURA","FREC_CARD","DURACION","CATEGORIA"]

print(f"\n  {'Campo':<15}  {'F1':>8}  {'P':>8}  {'R':>8}")
print("  " + "-"*44)
f1s = []
for c in CAMPOS:
    m = metricas_test.get(f"B-{c}", {})
    f = m.get("f",0)
    print(f"  {c:<15}  {f:>8.4f}  {m.get('p',0):>8.4f}  {m.get('r',0):>8.4f}")
    f1s.append(f)
macro_f1 = sum(f1s)/len(f1s)
print(f"\n  Macro F1 (B-): {macro_f1:.4f}")

modelo.save(MODEL_OUT)
print(f"\nModelo guardado: {MODEL_OUT}")

# ── Funciones de extraccion para comparativa ──────────────────────────────────
_TOKEN_RE_INF = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
SEXO_MAP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M","caballero":"M",
            "muchacho":"M","escolar":"M","adulto":"M",
            "femenino":"F","mujer":"F","femenina":"F","nina":"F","senora":"F","dama":"F",
            "muchacha":"F","escolar":"F","adulta":"F","lactante":"?"}
CAT_MAP  = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
            "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
            "vacunacion":"Vacunacion","nutricion":"Nutricion","embarazo":"Embarazo"}

def norm(s): return unicodedata.normalize("NFD",str(s)).encode("ascii","ignore").decode().lower().strip()
def to_int(v):
    try: return int(float(str(v).replace(",",".")))
    except: return None
def to_float(v):
    try: return round(float(str(v).replace(",",".")),1)
    except: return None

def extraer_keras(texto, modelo, vocab, label2id, id2label):
    tokens = [m.group() for m in _TOKEN_RE_INF.finditer(texto.lower())]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:MAX_LEN]]
    x = np.array([ids + [vocab["<PAD>"]]*(MAX_LEN-len(ids))], dtype=np.int32)
    pred_labels = [id2label[i] for i in modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]]
    res = {}
    for tok, lbl in zip(tokens, pred_labels):
        if lbl == "O": continue
        base = lbl[2:]
        if not lbl.startswith("B-"): continue
        if base=="EDAD" and "edad" not in res:
            v=to_int(tok)
            if v and 0<v<=120: res["edad"]=v
        elif base=="SEXO" and "sexo" not in res: res["sexo"]=SEXO_MAP.get(norm(tok))
        elif base=="PESO_KG" and "peso_kg" not in res: res["peso_kg"]=to_float(tok)
        elif base=="TALLA_CM" and "talla_cm" not in res:
            v=to_float(tok); res["talla_cm"]=round(v*100,1) if v and v<3 else v
        elif base=="PRESION_SIS" and "presion_sistolica" not in res: res["presion_sistolica"]=to_int(tok)
        elif base=="PRESION_DIA" and "presion_diastolica" not in res: res["presion_diastolica"]=to_int(tok)
        elif base=="GLUCOSA" and "glucosa_mg_dl" not in res: res["glucosa_mg_dl"]=to_int(tok)
        elif base=="TEMPERATURA" and "temperatura_c" not in res: res["temperatura_c"]=to_float(tok)
        elif base=="FREC_CARD" and "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"]=to_int(tok)
        elif base=="DURACION" and "duracion_sintomas_dias" not in res: res["duracion_sintomas_dias"]=to_int(tok)
        elif base=="CATEGORIA" and "categoria_sintoma" not in res: res["categoria_sintoma"]=CAT_MAP.get(norm(tok),tok)
    return res

def ok(pred, esp, campo):
    if pred is None: return False
    tol = {"edad":1,"peso_kg":1,"talla_cm":2,"presion_sistolica":1,"presion_diastolica":1,
           "glucosa_mg_dl":1,"temperatura_c":0.2,"frecuencia_cardiaca_bpm":1,"duracion_sintomas_dias":1}.get(campo)
    if tol: return abs(float(pred)-float(esp))<=tol
    return norm(str(pred))==norm(str(esp))

# ── SET 2: 55 casos de test_extractor.py ─────────────────────────────────────
# Textos ORIGINALES de test_extractor.py — sin modificar estructura ni contenido.
# Las tildes se mantienen; el extractor regex las normaliza internamente con _norm().
CASOS_55 = [
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

# ── SET 3: 20 frases ciegas escritas a mano ───────────────────────────────────
# NO derivadas de ninguna plantilla de entrenamiento
# NO derivadas de los 55 casos de referencia
CASOS_CIEGOS = [
    ("Ciego 01 — tension/calentura",
     "Tension arterial 128/82. Calentura de 37.9 grados. Glucemia 91 mg/dl. "
     "FC 76 lpm. Senora de 52 anos. Peso 68 kilos, estatura 160 cm. Desde hace 5 dias.",
     {"sexo":"F","edad":52,"presion_sistolica":128,"presion_diastolica":82,
      "temperatura_c":37.9,"glucosa_mg_dl":91,"frecuencia_cardiaca_bpm":76,
      "peso_kg":68,"talla_cm":160,"duracion_sintomas_dias":5}),
    ("Ciego 02 — muletilla pues",
     "Pues este paciente varon de 33 anos llego con 3 dias de evolucion. "
     "Peso 82 kg, talla 175 cm. Le tome la tension: 118/76 mmHg. "
     "Su azucar en sangre estaba 88. Temperatura 37.1. Frecuencia cardiaca 80.",
     {"sexo":"M","edad":33,"duracion_sintomas_dias":3,"peso_kg":82,"talla_cm":175,
      "presion_sistolica":118,"presion_diastolica":76,"glucosa_mg_dl":88,
      "temperatura_c":37.1,"frecuencia_cardiaca_bpm":80}),
    ("Ciego 03 — TA/Glx/T/FC",
     "Paciente femenino de 26 anos. Signos: TA 110/70, T 36.5, FC 72, Glx 85 mg/dl. "
     "Peso 55.5 kg, talla 1.58 m. Tiempo de evolucion: 7 dias.",
     {"sexo":"F","edad":26,"presion_sistolica":110,"presion_diastolica":70,
      "temperatura_c":36.5,"frecuencia_cardiaca_bpm":72,"glucosa_mg_dl":85,
      "peso_kg":55.5,"talla_cm":158,"duracion_sintomas_dias":7}),
    ("Ciego 04 — calentura/pulso/nivel glucosa",
     "Se atiende a caballero de 71 anos. Calentura de 38.1 grados. "
     "Presion 162/98 mmHg. Pulso 94 latidos. Nivel de glucosa 145. "
     "Peso 79 kg, mide 163 cm. Lleva una semana enfermo.",
     {"sexo":"M","edad":71,"temperatura_c":38.1,"presion_sistolica":162,
      "presion_diastolica":98,"frecuencia_cardiaca_bpm":94,"glucosa_mg_dl":145,
      "peso_kg":79,"talla_cm":163,"duracion_sintomas_dias":7}),
    ("Ciego 05 — pediatrico escolar masa/altura",
     "Doctor, le informo sobre el nino de 7 anos. Masa corporal 22 kg, altura 118 cm. "
     "Temperatura axilar 39.2 grados. Ritmo cardiaco 115 bpm. PA 90/55. Inicio hace 2 dias.",
     {"sexo":"M","edad":7,"peso_kg":22,"talla_cm":118,"temperatura_c":39.2,
      "frecuencia_cardiaca_bpm":115,"presion_sistolica":90,"presion_diastolica":55,
      "duracion_sintomas_dias":2}),
    ("Ciego 06 — glicemia ayunas/tabla datos",
     "Paciente masculino, 54 anos, 80.5 kilogramos, 172 centimetros. "
     "Tension: 140/88. Glicemia en ayunas: 118. Temperatura: 36.9. Pulso: 84. Cuadro de 4 dias.",
     {"sexo":"M","edad":54,"peso_kg":80.5,"talla_cm":172,"presion_sistolica":140,
      "presion_diastolica":88,"glucosa_mg_dl":118,"temperatura_c":36.9,
      "frecuencia_cardiaca_bpm":84,"duracion_sintomas_dias":4}),
    ("Ciego 07 — refiere/azucar/presion",
     "Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. "
     "Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78.",
     {"sexo":"F","edad":44,"duracion_sintomas_dias":9,"peso_kg":61,"talla_cm":155,
      "temperatura_c":37.3,"glucosa_mg_dl":97,"presion_sistolica":125,
      "presion_diastolica":80,"frecuencia_cardiaca_bpm":78}),
    ("Ciego 08 — sparse sin peso/talla",
     "Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, "
     "FC 68, glucosa 92. Cuadro agudo de 1 dia.",
     {"sexo":"M","edad":18,"presion_sistolica":108,"presion_diastolica":65,
      "temperatura_c":36.7,"frecuencia_cardiaca_bpm":68,"glucosa_mg_dl":92,
      "duracion_sintomas_dias":1}),
    ("Ciego 09 — entonces/tension alta coloquial",
     "Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. "
     "La tension le salio alta, en 175 sobre 105. "
     "La temperatura era de 37.4 y el pulso 88.",
     {"sexo":"F","edad":65,"duracion_sintomas_dias":10,"presion_sistolica":175,
      "presion_diastolica":105,"temperatura_c":37.4,"frecuencia_cardiaca_bpm":88}),
    ("Ciego 10 — pediatrico femenino",
     "Escolar femenino de 11 anos. Talla 140 cm, peso 36 kg. "
     "Temperatura de 38.6 grados. Pulso 98. Presion 95/60. Glucemia 88 mg/dl. 3 dias.",
     {"sexo":"F","edad":11,"talla_cm":140,"peso_kg":36,"temperatura_c":38.6,
      "frecuencia_cardiaca_bpm":98,"presion_sistolica":95,"presion_diastolica":60,
      "glucosa_mg_dl":88,"duracion_sintomas_dias":3}),
    ("Ciego 11 — llego con/estatura/azucar",
     "Hombre de 47 anos. Viene por cuadro de 8 dias. Peso actual: 91 kg. Estatura: 178 cm. "
     "Signos: tension 138/90, azucar 126 mg/dl, T 36.6C, latidos 82 por minuto.",
     {"sexo":"M","edad":47,"duracion_sintomas_dias":8,"peso_kg":91,"talla_cm":178,
      "presion_sistolica":138,"presion_diastolica":90,"glucosa_mg_dl":126,
      "temperatura_c":36.6,"frecuencia_cardiaca_bpm":82}),
    ("Ciego 12 — me llego/sin peso ni talla",
     "Me llego este paciente femenino de 29 anos, con fiebre de 39.5 y llevaba 4 dias asi. "
     "La presion la tenia en 105/68. El pulso en 102.",
     {"sexo":"F","edad":29,"temperatura_c":39.5,"duracion_sistomas_dias":4,
      "presion_sistolica":105,"presion_diastolica":68,"frecuencia_cardiaca_bpm":102}),
    ("Ciego 13 — datos completos con dos puntos",
     "Datos del paciente: edad 58 anos, sexo masculino. Medidas: 74 kg / 166 cm. "
     "Signos vitales: tension 148/92 mmHg, glucemia 210 mg/dl, temperatura 37.0 grados, "
     "FC 88 lpm. Evolucion: 6 dias.",
     {"edad":58,"sexo":"M","peso_kg":74,"talla_cm":166,"presion_sistolica":148,
      "presion_diastolica":92,"glucosa_mg_dl":210,"temperatura_c":37.0,
      "frecuencia_cardiaca_bpm":88,"duracion_sintomas_dias":6}),
    ("Ciego 14 — adulto mayor calentura/tension",
     "Adulto mayor de sexo femenino, 78 anos. Refiere inicio hace 2 dias. "
     "Peso 52 kg. Talla 149 cm. La calentura: 37.8. Azucar en ayunas: 135. "
     "Tension: 168/102. Latidos: 92.",
     {"sexo":"F","edad":78,"duracion_sintomas_dias":2,"peso_kg":52,"talla_cm":149,
      "temperatura_c":37.8,"glucosa_mg_dl":135,"presion_sistolica":168,
      "presion_diastolica":102,"frecuencia_cardiaca_bpm":92}),
    ("Ciego 15 — sparse sin peso/talla",
     "Paciente de 40 anos, sexo masculino. Sin datos de peso y talla. "
     "Tension arterial: 122/80. Temperatura: 38.0C. Glucosa: 99. "
     "Frecuencia cardiaca: 90. Evolucion de 3 dias.",
     {"edad":40,"sexo":"M","presion_sistolica":122,"presion_diastolica":80,
      "temperatura_c":38.0,"glucosa_mg_dl":99,"frecuencia_cardiaca_bpm":90,
      "duracion_sintomas_dias":3}),
    ("Ciego 16 — atiende medico/glucemia capilar",
     "Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. "
     "TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. "
     "Peso 67.5 kg. Estatura 1.74 m.",
     {"sexo":"M","edad":25,"duracion_sintomas_dias":5,"presion_sistolica":115,
      "presion_diastolica":72,"temperatura_c":36.4,"frecuencia_cardiaca_bpm":70,
      "glucosa_mg_dl":90,"peso_kg":67.5,"talla_cm":174}),
    ("Ciego 17 — dama/tension/nivel glucosa",
     "Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. "
     "Le registre tension de 120/78 mmHg. Nivel de glucosa 88. "
     "Temperatura axilar 37.2. Pulso 76 por minuto.",
     {"sexo":"F","edad":36,"duracion_sintomas_dias":11,"peso_kg":58,"talla_cm":162,
      "presion_sistolica":120,"presion_diastolica":78,"glucosa_mg_dl":88,
      "temperatura_c":37.2,"frecuencia_cardiaca_bpm":76}),
    ("Ciego 18 — pediatrico nino/tension/fiebre",
     "Nino de 5 anos, masculino. Fiebre 38.4C de 3 dias de evolucion. "
     "Peso 18 kg, mide 108 cm. Tension 88/58. FC 108. Glucosa 85.",
     {"sexo":"M","edad":5,"temperatura_c":38.4,"duracion_sintomas_dias":3,
      "peso_kg":18,"talla_cm":108,"presion_sistolica":88,"presion_diastolica":58,
      "frecuencia_cardiaca_bpm":108,"glucosa_mg_dl":85}),
    ("Ciego 19 — consulto/calentura/azucar/pesa",
     "Consulto paciente femenino de 62 anos con 14 dias de cuadro. "
     "Mide 1.53 m, pesa 71 kilos. Tension 158/96. Calentura 37.6. "
     "Azucar en sangre 142 mg/dl. Pulso 86 lpm.",
     {"sexo":"F","edad":62,"duracion_sintomas_dias":14,"talla_cm":153,"peso_kg":71,
      "presion_sistolica":158,"presion_diastolica":96,"temperatura_c":37.6,
      "glucosa_mg_dl":142,"frecuencia_cardiaca_bpm":86}),
    ("Ciego 20 — adolescente/tension x/temperatura/pulso",
     "Masculino, 15 anos. Talla 165 cm, peso 55 kg. Temperatura de 39.0 grados. "
     "Pulso 105. Tension arterial 100/65. Glucosa 90. Inicio hace 1 dia.",
     {"sexo":"M","edad":15,"talla_cm":165,"peso_kg":55,"temperatura_c":39.0,
      "frecuencia_cardiaca_bpm":105,"presion_sistolica":100,"presion_diastolica":65,
      "glucosa_mg_dl":90,"duracion_sintomas_dias":1}),
]

def evaluar_casos(casos, etiqueta):
    ok_total = 0; total = 0
    print(f"\n  {etiqueta}")
    print(f"  {'Caso':<30} {'Keras':>6}")
    print("  " + "-"*40)
    for nombre, texto, esperado in casos:
        res = extraer_keras(texto, modelo, VOCAB, LABEL2ID, ID2LABEL)
        n_ok = sum(1 for c,v in esperado.items() if ok(res.get(c),v,c))
        n    = len(esperado)
        ok_total += n_ok; total += n
        print(f"  {nombre:<30} {n_ok}/{n}")
    pct = ok_total/total*100 if total else 0
    print(f"  TOTAL: {ok_total}/{total}  ({pct:.1f}%)")
    return ok_total, total

print("\n" + "="*55)
print("  TABLA COMPARATIVA — 3 SETS DE EVALUACION")
print("="*55)

ok55, t55    = evaluar_casos(CASOS_55, "SET 2 — 55 casos de test_extractor.py (referencia)")
ok20, t20    = evaluar_casos(CASOS_CIEGOS, "SET 3 — 20 frases CIEGAS (generalizacion real)")

print("\n" + "="*55)
print(f"  SET 1 (sintetico v2):  Macro F1 = {macro_f1:.4f}")
print(f"  SET 2 (55 conocidos):  {ok55}/{t55}  ({ok55/t55*100:.1f}%)")
print(f"  SET 3 (20 ciegos):     {ok20}/{t20}  ({ok20/t20*100:.1f}%)")
print("="*55)

# Analisis de errores en set ciego
print("\n  ERRORES EN SET CIEGO (campos que fallan):")
error_campos = defaultdict(int)
for nombre, texto, esperado in CASOS_CIEGOS:
    res = extraer_keras(texto, modelo, VOCAB, LABEL2ID, ID2LABEL)
    for c, v in esperado.items():
        if not ok(res.get(c), v, c):
            error_campos[c] += 1
for campo, n_err in sorted(error_campos.items(), key=lambda x: -x[1]):
    print(f"    {campo:<30} {n_err} errores")

json.dump({
    "vocab_size":VOCAB_SIZE,"n_labels":N_LABELS,"max_len":MAX_LEN,
    "params":int(params),"epocas":len(historia.history["loss"]),
    "macro_f1_test_sintetico":round(macro_f1,4),
    "set2_55casos_pct":round(ok55/t55*100,1),
    "set3_20ciegos_pct":round(ok20/t20*100,1),
}, open(META_OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\nModelo v2 guardado: {MODEL_OUT}")
print("FASE C COMPLETA.")
