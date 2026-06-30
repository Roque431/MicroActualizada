"""
Verifica que la diferencia de etiquetas en C08 (Keras vs TFLite unrolled)
NO afecta los campos extraídos finales.
"""
import sys, io, json, re, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

with open("models/tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")
interp = tf.lite.Interpreter(model_path="models/ner_tflite_v2.tflite")
interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()

_RE = re.compile(r"\d+[.,]\d+|\d+|[a-z]+")
def tokenizar(t): return [m.group() for m in _RE.finditer(t.lower())][:40]
def norm(s): return unicodedata.normalize("NFD",str(s)).encode("ascii","ignore").decode().lower().strip()
def to_int(v):
    try: return int(float(str(v).replace(",",".")))
    except: return None
def to_float(v):
    try: return round(float(str(v).replace(",",".")),1)
    except: return None

SEXO_TOKENS = {"mujer","hombre","varon","masculino","femenino","femenina",
               "senora","senor","nina","nino","dama","caballero","muchacha","muchacho"}
DUR_CONTEXT = {"dias","dia","evolucion","cuadro","lleva","desde","hace"}
SEXO_MAP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M","caballero":"M",
            "mujer":"F","femenino":"F","femenina":"F","nina":"F","senora":"F","dama":"F"}
CAT_MAP  = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
            "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
            "vacunacion":"Vacunacion","nutricion":"Nutricion"}

def aplicar_reglas(tokens, labels):
    labels = list(labels)
    for i in range(len(labels)):
        if i < len(tokens) and labels[i]=="B-SEXO" and tokens[i].isdigit():
            hay_g = any(labels[j] in ("B-SEXO","I-SEXO") and tokens[j] in SEXO_TOKENS
                        for j in range(max(0,i-5),i))
            sig_a = i+1<len(tokens) and labels[i+1]=="I-SEXO" and tokens[i+1] in ("anos","an","a")
            if hay_g and sig_a: labels[i]="B-EDAD"; labels[i+1]="O"
    edad_idx = [i for i,l in enumerate(labels) if l=="B-EDAD"]
    if len(edad_idx) > 1:
        for idx in edad_idx[1:]:
            ini=max(0,idx-4); fin=min(len(tokens),idx+5)
            if set(tokens[ini:fin])-{tokens[idx]} & DUR_CONTEXT:
                labels[idx]="B-DURACION"
    return labels

def extraer(tokens, labels):
    res = {}
    for tok, lbl in zip(tokens, labels):
        if "SEXO" in lbl:
            v = SEXO_MAP.get(norm(tok))
            if v is not None: res["sexo"] = v; break
    for tok, lbl in zip(tokens, labels):
        if lbl == "O" or not lbl.startswith("B-"): continue
        base = lbl[2:]
        if base=="EDAD" and "edad" not in res:
            v = to_int(tok)
            if v and 0<v<=120: res["edad"] = v
        elif base=="PESO_KG" and "peso_kg" not in res:
            v = to_float(tok);
            if v is not None: res["peso_kg"] = v
        elif base=="TALLA_CM" and "talla_cm" not in res:
            v = to_float(tok)
            if v is not None: res["talla_cm"] = round(v*100,1) if v<3 else v
        elif base=="PRESION_SIS" and "presion_sistolica" not in res:
            v = to_int(tok)
            if v is not None: res["presion_sistolica"] = v
        elif base=="PRESION_DIA" and "presion_diastolica" not in res:
            v = to_int(tok)
            if v is not None: res["presion_diastolica"] = v
        elif base=="GLUCOSA" and "glucosa_mg_dl" not in res:
            v = to_int(tok)
            if v is not None: res["glucosa_mg_dl"] = v
        elif base=="TEMPERATURA" and "temperatura_c" not in res:
            v = to_float(tok)
            if v is not None: res["temperatura_c"] = v
        elif base=="FREC_CARD" and "frecuencia_cardiaca_bpm" not in res:
            v = to_int(tok)
            if v is not None: res["frecuencia_cardiaca_bpm"] = v
        elif base=="DURACION" and "duracion_sintomas_dias" not in res:
            v = to_int(tok)
            if v is not None: res["duracion_sintomas_dias"] = v
        elif base=="CATEGORIA" and "categoria_sintoma" not in res:
            v = CAT_MAP.get(norm(tok))
            if v is not None: res["categoria_sintoma"] = v
    return res

texto_c08 = ("Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, "
              "FC 68, glucosa 92. Cuadro agudo de 1 dia.")
esperado   = {"sexo":"M","edad":18,"presion_sistolica":108,"presion_diastolica":65,
              "temperatura_c":36.7,"frecuencia_cardiaca_bpm":68,"glucosa_mg_dl":92,
              "duracion_sintomas_dias":1}

tokens = tokenizar(texto_c08)
ids    = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
x      = np.array([ids + [0]*(40-len(ids))], dtype=np.int32)

# Keras
pred_k = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
labels_keras = [ID2LABEL[i] for i in pred_k]
labels_pp_keras = aplicar_reglas(tokens, labels_keras)
campos_keras = extraer(tokens, labels_pp_keras)

# TFLite
interp.set_tensor(inp_det[0]["index"], x)
interp.invoke()
logits = interp.get_tensor(out_det[0]["index"])[0]
pred_t = logits.argmax(axis=-1)[:len(tokens)]
labels_tflite = [ID2LABEL[i] for i in pred_t]
labels_pp_tflite = aplicar_reglas(tokens, labels_tflite)
campos_tflite = extraer(tokens, labels_pp_tflite)

print("C08:", texto_c08[:60], "...")
print(f"\nTokens   : {tokens}")
print(f"\nKeras labels     : {labels_keras}")
print(f"TFLite labels    : {labels_tflite}")

diffs_label = [(i, tokens[i], labels_keras[i], labels_tflite[i])
               for i in range(len(tokens)) if labels_keras[i] != labels_tflite[i]]
print(f"\nDiferencias de etiqueta ({len(diffs_label)}):")
for i, tok, k, t in diffs_label:
    print(f"  [{i}] '{tok}': Keras={k}  TFLite={t}")

print(f"\nKeras labels_pp  : {labels_pp_keras}")
print(f"TFLite labels_pp : {labels_pp_tflite}")

print(f"\nKeras  campos: {campos_keras}")
print(f"TFLite campos: {campos_tflite}")
print(f"Esperado:      {esperado}")

campos_iguales = campos_keras == campos_tflite
print(f"\nCampos extraídos idénticos: {'SI' if campos_iguales else 'NO'}")
ok_keras  = {c for c,v in esperado.items() if campos_keras.get(c)==v}
ok_tflite = {c for c,v in esperado.items() if campos_tflite.get(c)==v}
print(f"Campos correctos Keras : {len(ok_keras)}/{len(esperado)}")
print(f"Campos correctos TFLite: {len(ok_tflite)}/{len(esperado)}")
