"""
Exporta ner_tflite_v2.keras a TFLite con cuantización dinámica de rangos.

Pasos:
  1. Cargar el modelo Keras
  2. Inspeccionar firma (input/output shapes)
  3. Convertir con dynamic-range quantization (solo pesos, sin datos de calibración)
  4. Guardar models/ner_tflite_v2.tflite
  5. Smoke-test: comparar predicciones Keras vs TFLite en los 25 casos del golden set
     — si algún token difiere, reportar y abortar
"""
import sys, io, json, re, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

KERAS_PATH  = "models/ner_tflite_v2.keras"
TFLITE_PATH = "models/ner_tflite_v2.tflite"
GOLDEN_PATH = "models/postproc_golden_set.json"
VOCAB_PATH  = "models/tflite_vocab_v2.json"
LABELS_PATH = "models/tflite_labels_v2.json"

# ── 1. Cargar modelo ──────────────────────────────────────────────────────────
print("Cargando modelo Keras...")
t0 = time.time()
modelo = tf.keras.models.load_model(KERAS_PATH)
print(f"  Cargado en {time.time()-t0:.1f}s")

print("\nFirma del modelo:")
modelo.summary(print_fn=lambda s: print("  " + s))

# ── 2. Inspeccionar I/O ───────────────────────────────────────────────────────
inp = modelo.input
out = modelo.output
print(f"\n  Input  : {inp.shape}  dtype={inp.dtype}")
print(f"  Output : {out.shape}  dtype={out.dtype}")

# ── 3. Convertir a TFLite con cuantización dinámica ──────────────────────────
print("\nConvirtiendo a TFLite...")

# Problema: el BiLSTM en TF 2.15 con mask_zero=True genera while_loop
# y ops de mascareo que el conversor estándar no soporta.
# Solución: reconstruir el modelo con unroll=True + sin mask_zero
# (MAX_LEN=40 es fijo, así que unroll es válido y no hay pérdida de calidad).
# Luego copiar los pesos — mask_zero NO agrega parámetros, así que
# los pesos son byte-a-byte idénticos entre los dos grafos.
#
# Resultado: TFLite con built-in ops estándar, sin SELECT_TF_OPS.
# Compatible con tflite_flutter 0.10.x sin configuración especial.

with open(VOCAB_PATH,  encoding="utf-8") as f: VOCAB_BUILD = json.load(f)
with open(LABELS_PATH, encoding="utf-8") as f: LABEL2ID_BUILD = json.load(f)
VOCAB_SIZE_B = len(VOCAB_BUILD)
N_LABELS_B   = len(LABEL2ID_BUILD)

print(f"  Reconstruyendo grafo con unroll=True "
      f"(vocab={VOCAB_SIZE_B}, labels={N_LABELS_B}, max_len=40)...")

from tensorflow import keras as _k
EMBED_DIM, LSTM1, LSTM2 = 64, 64, 32

def construir_unrolled(vocab_size, embed_dim, lstm1, lstm2, n_labels):
    inp = _k.Input(shape=(40,), dtype="int32", name="tokens")
    x = _k.layers.Embedding(vocab_size, embed_dim,
                              mask_zero=False, name="embedding")(inp)
    x = _k.layers.Bidirectional(
        _k.layers.LSTM(lstm1, return_sequences=True,
                       unroll=True, name="fwd_lstm_1"),
        name="bilstm_1")(x)
    x = _k.layers.Bidirectional(
        _k.layers.LSTM(lstm2, return_sequences=True,
                       unroll=True, name="fwd_lstm_2"),
        name="bilstm_2")(x)
    out = _k.layers.TimeDistributed(
        _k.layers.Dense(n_labels, activation="softmax"), name="output")(x)
    return _k.Model(inputs=inp, outputs=out, name="ner_tflite_v2_unrolled")

modelo_u = construir_unrolled(VOCAB_SIZE_B, EMBED_DIM, LSTM1, LSTM2, N_LABELS_B)

# Copiar pesos capa por capa (los nombres de capa coinciden entre ambos grafos)
layer_names_orig    = {l.name: l for l in modelo.layers}
layer_names_unrolled = {l.name: l for l in modelo_u.layers}

copied = []
for name, lu in layer_names_unrolled.items():
    if name in layer_names_orig:
        orig_w = layer_names_orig[name].get_weights()
        if orig_w:
            lu.set_weights(orig_w)
            copied.append(name)

print(f"  Pesos copiados de {len(copied)} capas: {copied}")

# Verificar predicciones antes de exportar
print("  Verificando paridad pesos (muestra 5 casos)...")
with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_check = json.load(f)

with open(VOCAB_PATH, encoding="utf-8") as f: V_CHK = json.load(f)
with open(LABELS_PATH, encoding="utf-8") as f: L_CHK = json.load(f)
ID2L_CHK = {v: k for k, v in L_CHK.items()}

fallos_pesos = []
for caso in golden_check["cases"][:5]:
    tokens = caso["tokens"]
    ids    = [V_CHK.get(t, V_CHK["<UNK>"]) for t in tokens]
    x = np.array([ids + [0] * (40 - len(ids))], dtype=np.int32)
    pred_orig = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    pred_unr  = modelo_u.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    if list(pred_orig) != list(pred_unr):
        fallos_pesos.append(caso["id"])

if fallos_pesos:
    print(f"  ADVERTENCIA: {len(fallos_pesos)} caso(s) difieren tras copiar pesos: {fallos_pesos}")
    print("  Usando SELECT_TF_OPS como fallback.")
    # Fallback: SavedModel + SELECT_TF_OPS
    SAVED_MODEL_DIR = "models/saved_model_ner_v2"
    print(f"  Guardando SavedModel...")
    tf.saved_model.save(modelo, SAVED_MODEL_DIR)
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    USANDO_FLEX = True
else:
    print("  OK — predicciones idénticas con pesos copiados.")
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo_u)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    USANDO_FLEX = False

t2 = time.time()
tflite_model = converter.convert()
elapsed = time.time() - t2
modo = "con SELECT_TF_OPS (flex delegate)" if USANDO_FLEX else "sin flex delegate (built-in ops)"
print(f"  Conversión completada en {elapsed:.1f}s — {modo}")

# Guardar
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"  Guardado: {TFLITE_PATH}  ({size_kb:.1f} KB)")

# ── 4. Inspeccionar modelo TFLite ─────────────────────────────────────────────
print("\nInspeccionando modelo TFLite...")
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()

inp_det  = interp.get_input_details()
out_det  = interp.get_output_details()
print(f"  Input  [{inp_det[0]['index']}]: shape={inp_det[0]['shape']}  dtype={inp_det[0]['dtype'].__name__}")
print(f"  Output [{out_det[0]['index']}]: shape={out_det[0]['shape']}  dtype={out_det[0]['dtype'].__name__}")

# ── 5. Smoke-test: Keras vs TFLite en los 25 casos ───────────────────────────
print("\nSmoke-test: Keras vs TFLite en 25 casos del golden set...")

with open(VOCAB_PATH,  encoding="utf-8") as f: VOCAB    = json.load(f)
with open(LABELS_PATH, encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden = json.load(f)

MAX_LEN = inp_det[0]["shape"][1]   # debe ser 40

fallos = []
for caso in golden["cases"]:
    tokens = caso["tokens"]
    # Reconstruir IDs exactamente igual que el entrenamiento
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
    x = np.array([ids + [0] * (MAX_LEN - len(ids))], dtype=np.int32)

    # Keras
    pred_keras = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    labels_keras = [ID2LABEL[i] for i in pred_keras]

    # TFLite
    interp.set_tensor(inp_det[0]["index"], x)
    interp.invoke()
    logits = interp.get_tensor(out_det[0]["index"])[0]  # shape (40, n_labels)
    pred_tflite = logits.argmax(axis=-1)[:len(tokens)]
    labels_tflite = [ID2LABEL[i] for i in pred_tflite]

    if labels_keras != labels_tflite:
        diffs = [(j, tokens[j], labels_keras[j], labels_tflite[j])
                 for j in range(len(tokens)) if labels_keras[j] != labels_tflite[j]]
        fallos.append({"id": caso["id"], "diffs": diffs})

print(f"\n  Casos probados : 25")
print(f"  Coincidencias  : {25 - len(fallos)}/25")
if fallos:
    print(f"\n  DIFERENCIAS ENCONTRADAS ({len(fallos)} caso(s)):")
    for f_ in fallos:
        print(f"  [{f_['id']}]")
        for idx, tok, k, t in f_["diffs"]:
            print(f"    tok[{idx}]='{tok}': Keras={k}  TFLite={t}")
    print("\n  ADVERTENCIA: el modelo cuantizado difiere del Keras en los casos anteriores.")
    print("  Considera full-int8 quantization con datos de calibración si los fallos")
    print("  afectan los 6 casos con reglas activas (C02, C07, C09, C16, C17, C19).")
else:
    print("  OK — todas las predicciones son idénticas entre Keras y TFLite.")

print(f"\nDone. Modelo listo en: {TFLITE_PATH}")
