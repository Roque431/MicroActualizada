import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, tensorflow as tf
tf.get_logger().setLevel("ERROR")

_RE = re.compile(r"\d+[.,]\d+|\d+|[a-z]+")
def tokenizar(t): return [m.group() for m in _RE.finditer(t.lower())]

with open("models/tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v:k for k,v in LABEL2ID.items()}
modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")

def raw_labels(texto):
    tokens = tokenizar(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens[:40]]
    x = np.array([ids + [0]*(40-len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    return tokens, [ID2LABEL[i] for i in pred]

SEXO_TOKENS = {"mujer","hombre","varon","masculino","femenino","femenina",
               "senora","senor","nina","nino","dama","caballero","muchacha","muchacho"}
DUR_CONTEXT = {"dias","dia","evolucion","cuadro","lleva","desde","hace"}

def aplicar_reglas(tokens, labels):
    labels = list(labels)
    for i in range(len(labels)):
        if i < len(tokens) and labels[i] == "B-SEXO" and tokens[i].isdigit():
            hay_genero = any(labels[j] in ("B-SEXO","I-SEXO") and tokens[j] in SEXO_TOKENS
                             for j in range(max(0,i-5),i))
            sig_anos = (i+1<len(tokens) and labels[i+1]=="I-SEXO" and tokens[i+1] in ("anos","an","a"))
            if hay_genero and sig_anos:
                labels[i] = "B-EDAD"; labels[i+1] = "O"
    edad_idx = [i for i,l in enumerate(labels) if l=="B-EDAD"]
    if len(edad_idx) > 1:
        for idx in edad_idx[1:]:
            ini=max(0,idx-4); fin=min(len(tokens),idx+5)
            ctx=set(tokens[ini:fin])-{tokens[idx]}
            if ctx & DUR_CONTEXT: labels[idx]="B-DURACION"
    return labels

def extraer_duracion_SIN_guarda(tokens, labels):
    """Extractor BUGGY: acepta B-DURACION aunque to_int devuelva None."""
    for tok, lbl in zip(tokens, labels):
        if lbl == "B-DURACION":
            try: return int(tok)
            except: return None  # <- None se guarda, bloquea siguiente token
    return None

def extraer_duracion_CON_guarda(tokens, labels):
    """Extractor CORRECTO: salta B-DURACION si el token no es numerico."""
    for tok, lbl in zip(tokens, labels):
        if lbl == "B-DURACION":
            try:
                v = int(tok)
                return v          # <- solo retorna si es un numero valido
            except:
                continue          # <- texto no numerico: siguiente B-DURACION
    return None

CASOS = [
    ("Ciego 09",
     "Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. "
     "La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88.",
     10),
    ("Ciego 16",
     "Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. "
     "TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. "
     "Peso 67.5 kg. Estatura 1.74 m.",
     5),
]

print(f"{'Caso':<12}  {'Sin reglas':>10}  {'Con reglas (bug)':>18}  {'Con reglas + guarda':>21}  {'Esperado':>10}  Correcto?")
print("-"*85)
for nombre, texto, esperado in CASOS:
    tokens, labels_raw = raw_labels(texto)
    labels_pp = aplicar_reglas(tokens, labels_raw)

    # 1. Sin reglas
    sin_reglas = extraer_duracion_SIN_guarda(tokens, labels_raw)

    # 2. Con reglas pero SIN guarda (el bug que causaba la regresion)
    con_reglas_bug = extraer_duracion_SIN_guarda(tokens, labels_pp)

    # 3. Con reglas Y CON guarda (version correcta actual)
    con_reglas_ok = extraer_duracion_CON_guarda(tokens, labels_pp)

    correcto = "SI" if con_reglas_ok == esperado else "NO"
    print(f"{nombre:<12}  {str(sin_reglas):>10}  {str(con_reglas_bug):>18}  {str(con_reglas_ok):>21}  {str(esperado):>10}  {correcto}")

    # Detalle de los B-DURACION para mostrar exactamente que tokens estan etiquetados
    b_dur_pp = [(tok, lbl) for tok, lbl in zip(tokens, labels_pp) if lbl == "B-DURACION"]
    print(f"             B-DURACION tras reglas: {b_dur_pp}")
    print()
