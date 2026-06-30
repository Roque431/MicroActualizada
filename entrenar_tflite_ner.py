"""
FASE 2 — Entrenamiento del modelo NER en Keras para TFLite.

Arquitectura: Embedding -> BiLSTM -> BiLSTM (pequeña) -> Dense (softmax por token)
Disenada para pesar < 5 MB como .tflite y correr en milisegundos en gama media-baja.

Ejecutar:
    python entrenar_tflite_ner.py
"""
import os, json, random, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Configuracion ─────────────────────────────────────────────────────────────
DATA_DIR   = Path("models")
MODEL_H5   = DATA_DIR / "ner_tflite.keras"
META_PATH  = DATA_DIR / "ner_tflite_meta.json"
MAX_LEN    = 40      # longitud maxima determinada en Fase 1
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 0.001
EMBED_DIM  = 64
LSTM1      = 64      # unidades por direccion en la primera BiLSTM
LSTM2      = 32      # unidades por direccion en la segunda BiLSTM

print(f"TensorFlow {tf.__version__}")

# ── Cargar vocabulario y etiquetas (generados en Fase 1) ─────────────────────
with open(DATA_DIR / "tflite_vocab.json",  encoding="utf-8") as f:
    VOCAB   = json.load(f)
with open(DATA_DIR / "tflite_labels.json", encoding="utf-8") as f:
    LABEL2ID = json.load(f)

ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
VOCAB_SIZE = len(VOCAB)
N_LABELS   = len(LABEL2ID)
PAD_ID     = VOCAB["<PAD>"]
O_ID       = LABEL2ID["O"]

print(f"Vocab: {VOCAB_SIZE}  |  Etiquetas: {N_LABELS}  |  MAX_LEN: {MAX_LEN}")

# ── Cargar y padear dataset ───────────────────────────────────────────────────
def cargar_y_padear(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    X, y = [], []
    for ej in data:
        ids    = ej["token_ids"][:MAX_LEN]
        labels = ej["label_ids"][:MAX_LEN]
        pad_n  = MAX_LEN - len(ids)
        X.append(ids    + [PAD_ID] * pad_n)
        y.append(labels + [O_ID]   * pad_n)   # O (0) es la etiqueta de padding
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

print("\nCargando datasets...")
X_train, y_train = cargar_y_padear(DATA_DIR / "tflite_train.json")
X_val,   y_val   = cargar_y_padear(DATA_DIR / "tflite_val.json")
X_test,  y_test  = cargar_y_padear(DATA_DIR / "tflite_test.json")
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── Class weights (compensar el imbalance O vs entidades) ────────────────────
# O es ~60% del dataset; ponderamos las entidades para que el modelo las aprenda
label_counts = np.bincount(y_train.flatten(), minlength=N_LABELS)
total_tokens = label_counts.sum()
class_weight = {}
for lid in range(N_LABELS):
    if lid == O_ID:
        class_weight[lid] = 0.3  # penalizar menos el O
    else:
        # peso inversamente proporcional a la frecuencia
        freq = label_counts[lid] / total_tokens
        class_weight[lid] = min(1.0 / (freq * N_LABELS), 5.0)  # cap en 5x

print(f"\n  Class weights: O={class_weight[O_ID]:.2f}  "
      f"B-EDAD={class_weight[LABEL2ID['B-EDAD']]:.2f}  "
      f"B-SEXO={class_weight[LABEL2ID['B-SEXO']]:.2f}")

# ── Arquitectura Keras ────────────────────────────────────────────────────────
# Embedding + 2x BiLSTM + Dense softmax por token
# Intencionalmente simple: soportada al 100% por TFLite
def construir_modelo():
    inp = keras.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")

    # Embedding — mask_zero para ignorar el padding en la LSTM
    x = keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="embedding"
    )(inp)

    # BiLSTM 1 — captura contexto bidireccional largo
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(LSTM1, return_sequences=True, dropout=0.2),
        name="bilstm_1"
    )(x)
    x = keras.layers.Dropout(0.3, name="dropout_1")(x)

    # BiLSTM 2 — refina con contexto mas local
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(LSTM2, return_sequences=True, dropout=0.1),
        name="bilstm_2"
    )(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)

    # Clasificacion por token
    out = keras.layers.TimeDistributed(
        keras.layers.Dense(N_LABELS, activation="softmax"),
        name="output"
    )(x)

    return keras.Model(inputs=inp, outputs=out, name="ner_tflite")

modelo = construir_modelo()
modelo.summary()

# Estimar tamano en MB
params = modelo.count_params()
print(f"\n  Parametros totales: {params:,}")
print(f"  Tamano estimado (fp32): {params * 4 / 1024**2:.2f} MB")
print(f"  Tamano estimado (int8 quantizado): {params / 1024**2:.2f} MB")

# ── Compilar ──────────────────────────────────────────────────────────────────
modelo.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    ),
]

# ── Entrenamiento ─────────────────────────────────────────────────────────────
print(f"\nEntrenando: {EPOCHS} epocas max, batch={BATCH_SIZE}")
historia = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluacion por entidad (F1, Precision, Recall) ───────────────────────────
def calcular_f1_por_entidad(X, y_true, modelo, id2label):
    y_pred = modelo.predict(X, verbose=0).argmax(axis=-1)  # (n, MAX_LEN)

    # TP, FP, FN por etiqueta (ignorando O y padding)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for seq_pred, seq_true in zip(y_pred, y_true):
        for pred_id, true_id in zip(seq_pred, seq_true):
            true_lbl = id2label[true_id]
            pred_lbl = id2label[pred_id]

            if true_lbl == "O" and pred_lbl == "O":
                continue
            if true_lbl != "O":
                if pred_lbl == true_lbl:
                    tp[true_lbl] += 1
                else:
                    fn[true_lbl] += 1
                    if pred_lbl != "O":
                        fp[pred_lbl] += 1
            else:  # true == O, pred != O
                fp[pred_lbl] += 1

    resultados = {}
    for lbl in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
        p = tp[lbl] / (tp[lbl] + fp[lbl] + 1e-9)
        r = tp[lbl] / (tp[lbl] + fn[lbl] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        resultados[lbl] = {"p": round(p, 4), "r": round(r, 4), "f": round(f, 4),
                           "tp": tp[lbl], "fp": fp[lbl], "fn": fn[lbl]}
    return resultados

print("\nEvaluando en TEST SET...")
metricas = calcular_f1_por_entidad(X_test, y_test, modelo, ID2LABEL)

# Agrupar B- e I- por campo base
CAMPOS = ["EDAD", "SEXO", "PESO_KG", "TALLA_CM", "PRESION_SIS", "PRESION_DIA",
          "GLUCOSA", "TEMPERATURA", "FREC_CARD", "DURACION", "CATEGORIA"]

print("\n=== METRICAS EN TEST SET ===")
print(f"  {'Campo':<15}  {'F1(B-)':>8}  {'P(B-)':>8}  {'R(B-)':>8}")
print("  " + "-"*46)
f1_vals = []
for campo in CAMPOS:
    b_key = f"B-{campo}"
    if b_key in metricas:
        m = metricas[b_key]
        print(f"  {campo:<15}  {m['f']:>8.4f}  {m['p']:>8.4f}  {m['r']:>8.4f}")
        f1_vals.append(m['f'])
    else:
        print(f"  {campo:<15}  {'N/A':>8}")
macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0
print(f"\n  Macro F1 (B-): {macro_f1:.4f}")

# ── Guardar modelo ────────────────────────────────────────────────────────────
modelo.save(MODEL_H5)
print(f"\nModelo Keras guardado: {MODEL_H5}")

meta = {
    "vocab_size": VOCAB_SIZE,
    "n_labels": N_LABELS,
    "max_len": MAX_LEN,
    "embed_dim": EMBED_DIM,
    "lstm1_units": LSTM1 * 2,
    "lstm2_units": LSTM2 * 2,
    "total_params": int(params),
    "epochs_entrenados": len(historia.history["loss"]),
    "mejor_val_loss": float(min(historia.history["val_loss"])),
    "macro_f1_test": round(macro_f1, 4),
    "f1_por_campo": {c: metricas.get(f"B-{c}", {}).get("f", 0) for c in CAMPOS},
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"Metricas guardadas: {META_PATH}")

# ── Tabla comparativa con los 55 casos de test_extractor.py ──────────────────
print("\n\n=== TABLA COMPARATIVA — 55 casos de test_extractor.py ===")
print("    (regex vs spaCy vs Keras/TFLite)")

import re, unicodedata, sys
sys.path.insert(0, str(Path(__file__).parent))

# Importar el extractor regex desde prueba_voz.py
import importlib.util as ilu
spec2 = ilu.spec_from_file_location("prueba_voz", "prueba_voz.py")
pv = ilu.module_from_spec(spec2)
spec2.loader.exec_module(pv)
extraer_regex = pv.extraer_variables_local

# Importar el modelo spaCy
import spacy
nlp_spacy = spacy.load("models/ner_clinico_es")
SEXO_MAP_SP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M",
               "femenino":"F","mujer":"F","femenina":"F","nina":"F","senora":"F"}
CAT_MAP_SP = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
              "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
              "vacunacion":"Vacunacion","nutricion":"Nutricion","embarazo":"Embarazo"}

def norm(s): return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower().strip()
def to_int(v):
    try: return int(float(str(v).replace(",",".")))
    except: return None
def to_float(v):
    try: return round(float(str(v).replace(",",".")), 1)
    except: return None

def extraer_spacy(texto):
    doc = nlp_spacy(texto)
    res = {}
    for e in doc.ents:
        t, l = e.text.strip(), e.label_
        if l=="EDAD" and "edad" not in res:
            v=to_int(t)
            if v and 0<v<=120: res["edad"]=v
        elif l=="SEXO" and "sexo" not in res: res["sexo"]=SEXO_MAP_SP.get(norm(t))
        elif l=="PESO_KG" and "peso_kg" not in res: res["peso_kg"]=to_float(t)
        elif l=="TALLA_CM" and "talla_cm" not in res:
            v=to_float(t); res["talla_cm"]=round(v*100,1) if v and v<3 else v
        elif l=="PRESION_SIS" and "presion_sistolica" not in res:
            if "/" in t:
                p=t.split("/"); res["presion_sistolica"]=to_int(p[0]); res["presion_diastolica"]=to_int(p[1])
            else: res["presion_sistolica"]=to_int(t)
        elif l=="PRESION_DIA" and "presion_diastolica" not in res: res["presion_diastolica"]=to_int(t)
        elif l=="GLUCOSA" and "glucosa_mg_dl" not in res: res["glucosa_mg_dl"]=to_int(t)
        elif l=="TEMPERATURA" and "temperatura_c" not in res: res["temperatura_c"]=to_float(t)
        elif l=="FREC_CARD":
            if "/" in t and "presion_sistolica" not in res:
                p=t.split("/")
                try:
                    ps,pd=int(p[0]),int(p[1])
                    if 60<=ps<=250 and 40<=pd<=150: res["presion_sistolica"]=ps; res["presion_diastolica"]=pd; continue
                except: pass
            if "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"]=to_int(t)
        elif l=="DURACION" and "duracion_sintomas_dias" not in res: res["duracion_sintomas_dias"]=to_int(t)
        elif l=="CATEGORIA" and "categoria_sintoma" not in res: res["categoria_sintoma"]=CAT_MAP_SP.get(norm(t),t)
    return res

def extraer_keras(texto):
    """Inferencia con el modelo Keras."""
    _TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
    tokens = [m.group() for m in _TOKEN_RE.finditer(texto.lower())]
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens[:MAX_LEN]]
    pad_n = MAX_LEN - len(ids)
    x = np.array([ids + [PAD_ID]*pad_n], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0]
    pred_labels = [ID2LABEL[i] for i in pred.argmax(axis=-1)[:len(tokens)]]

    res = {}
    for tok, lbl in zip(tokens, pred_labels):
        if lbl == "O": continue
        campo_raw = lbl[2:]  # quitar B- o I-
        if lbl.startswith("B-"):
            if campo_raw == "EDAD" and "edad" not in res:
                v=to_int(tok)
                if v and 0<v<=120: res["edad"]=v
            elif campo_raw == "SEXO" and "sexo" not in res: res["sexo"]=SEXO_MAP_SP.get(norm(tok))
            elif campo_raw == "PESO_KG" and "peso_kg" not in res: res["peso_kg"]=to_float(tok)
            elif campo_raw == "TALLA_CM" and "talla_cm" not in res:
                v=to_float(tok); res["talla_cm"]=round(v*100,1) if v and v<3 else v
            elif campo_raw == "PRESION_SIS" and "presion_sistolica" not in res: res["presion_sistolica"]=to_int(tok)
            elif campo_raw == "PRESION_DIA" and "presion_diastolica" not in res: res["presion_diastolica"]=to_int(tok)
            elif campo_raw == "GLUCOSA" and "glucosa_mg_dl" not in res: res["glucosa_mg_dl"]=to_int(tok)
            elif campo_raw == "TEMPERATURA" and "temperatura_c" not in res: res["temperatura_c"]=to_float(tok)
            elif campo_raw == "FREC_CARD" and "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"]=to_int(tok)
            elif campo_raw == "DURACION" and "duracion_sintomas_dias" not in res: res["duracion_sintomas_dias"]=to_int(tok)
            elif campo_raw == "CATEGORIA" and "categoria_sintoma" not in res: res["categoria_sintoma"]=CAT_MAP_SP.get(norm(tok),tok)
    return res

# Los mismos 5 casos de test_extractor.py  (11 campos cada uno = 55 total)
CASOS = [
    ("SOAP clasico",
     "Consulta medica. Paciente masculino de 45 anos de edad. "
     "Peso 82.0 kg, talla 170.5 cm. "
     "Presion arterial de 125/82 mmHg, glucosa de 98 mg/dl, "
     "temperatura 37.2C, frecuencia cardiaca de 91 latidos por minuto. "
     "5 dias de evolucion.",
     {"edad":45,"sexo":"M","peso_kg":82.0,"talla_cm":170.5,
      "presion_sistolica":125,"presion_diastolica":82,
      "glucosa_mg_dl":98,"temperatura_c":37.2,"frecuencia_cardiaca_bpm":91,"duracion_sintomas_dias":5}),
    ("Dictado rapido",
     "Paciente femenina, 67 anos. PA: 160/95, glucemia capilar 310 mg/dL. "
     "T: 36.8 grados. FC: 88 lpm. Peso 63.5 kg, estatura de 155.0 cm. "
     "Sintomas desde hace 3 dias.",
     {"edad":67,"sexo":"F","peso_kg":63.5,"talla_cm":155.0,
      "presion_sistolica":160,"presion_diastolica":95,
      "glucosa_mg_dl":310,"temperatura_c":36.8,"frecuencia_cardiaca_bpm":88,"duracion_sintomas_dias":3}),
    ("Coloquial comunidad",
     "Se atiende a mujer de 32 anos que lleva 7 dias con tos y dificultad para respirar. "
     "Presion 118 sobre 74, azucar en sangre 89, temperatura 38.5, pulso 102. "
     "Pesa 58 kilos, mide 162 cm.",
     {"edad":32,"sexo":"F","peso_kg":58.0,"talla_cm":162.0,
      "presion_sistolica":118,"presion_diastolica":74,
      "glucosa_mg_dl":89,"temperatura_c":38.5,"frecuencia_cardiaca_bpm":102,"duracion_sintomas_dias":7}),
    ("Pediatrico vacunacion",
     "Paciente masculino, edad: 2 anos. Consulta por fiebre leve postvacuna "
     "y control de esquema de vacunacion con 1 dia de evolucion. "
     "Peso corporal: 12.5 kg, talla 82.0 centimetros. "
     "Tension arterial 100/65 mmHg, temperatura de 37.8 grados, "
     "frecuencia cardiaca de 110 lpm, glucosa 88 mg/dl.",
     {"edad":2,"sexo":"M","peso_kg":12.5,"talla_cm":82.0,
      "presion_sistolica":100,"presion_diastolica":65,
      "glucosa_mg_dl":88,"temperatura_c":37.8,"frecuencia_cardiaca_bpm":110,"duracion_sintomas_dias":1}),
    ("Texto libre sin estructura",
     "La senora tiene como 55 anos, es hipertensa, refiere zumbido en los oidos "
     "y vision borrosa desde hace 4 dias. Le tome la presion y salio 170 sobre 100. "
     "Su azucar estaba bien, 95. Temperatura normal 36.5. Pulso 78. "
     "Pesa como 70 kilos y mide 158 centimetros.",
     {"edad":55,"sexo":"F","peso_kg":70.0,"talla_cm":158.0,
      "presion_sistolica":170,"presion_diastolica":100,
      "glucosa_mg_dl":95,"temperatura_c":36.5,"frecuencia_cardiaca_bpm":78,"duracion_sintomas_dias":4}),
]

TOLERANCIAS = {"edad":1,"peso_kg":1,"talla_cm":2,"presion_sistolica":1,"presion_diastolica":1,
               "glucosa_mg_dl":1,"temperatura_c":0.2,"frecuencia_cardiaca_bpm":1,"duracion_sintomas_dias":1}

def ok(pred, esp, campo):
    if pred is None: return False
    tol = TOLERANCIAS.get(campo)
    if tol: return abs(float(pred) - float(esp)) <= tol
    return norm(str(pred)) == norm(str(esp))

totales = {"regex":0,"spacy":0,"keras":0,"total":0}

print(f"\n  {'Caso':<25} {'regex':>6} {'spaCy':>6} {'Keras':>6}")
print("  " + "-"*47)

for nombre, texto, esperado in CASOS:
    r_regex = extraer_regex(texto)
    r_spacy = extraer_spacy(texto)
    r_keras = extraer_keras(texto)

    ok_r = sum(1 for c,v in esperado.items() if ok(r_regex.get(c), v, c))
    ok_s = sum(1 for c,v in esperado.items() if ok(r_spacy.get(c), v, c))
    ok_k = sum(1 for c,v in esperado.items() if ok(r_keras.get(c), v, c))
    n    = len(esperado)

    totales["regex"] += ok_r
    totales["spacy"] += ok_s
    totales["keras"] += ok_k
    totales["total"] += n

    print(f"  {nombre:<25} {ok_r}/{n}    {ok_s}/{n}    {ok_k}/{n}")

t = totales["total"]
print("  " + "-"*47)
print(f"  {'TOTAL (55 campos)':<25} "
      f"{totales['regex']}/{t}   "
      f"{totales['spacy']}/{t}   "
      f"{totales['keras']}/{t}")
print(f"\n  Exactitud:  "
      f"regex={totales['regex']/t*100:.1f}%  "
      f"spaCy={totales['spacy']/t*100:.1f}%  "
      f"Keras={totales['keras']/t*100:.1f}%")

print("\n\nFASE 2 COMPLETA.")
print("Si los resultados son aceptables, confirma para pasar a Fase 3 (exportar TFLite).")
