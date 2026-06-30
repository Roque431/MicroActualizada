import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, tensorflow as tf
tf.get_logger().setLevel("ERROR")

_RE = re.compile(r"\d+[.,]\d+|\d+|[a-z]+")
def tokenizar(t): return [m.group() for m in _RE.finditer(t.lower())]

with open("models/tflite_vocab_v2.json", encoding="utf-8") as f: VOCAB = json.load(f)
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
VENTANA = 4

casos = [
    ("Ciego 09",
     "Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. "
     "La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88.",
     65, 10),
    ("Ciego 16",
     "Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. "
     "TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. "
     "Peso 67.5 kg. Estatura 1.74 m.",
     25, 5),
]

for nombre, texto, edad_esp, dur_esp in casos:
    tokens, labels = raw_labels(texto)
    labels2 = list(labels)

    print(f"\n{'='*60}  {nombre}")

    # Labels relevantes del modelo
    print("  Labels no-O del modelo RAW:")
    for i,(t,l) in enumerate(zip(tokens,labels)):
        if l != "O": print(f"    [{i:>2}] {t:<12} {l}")

    # Paso 1: Regla 2
    print("\n  Regla 2 — B-SEXO en token numerico:")
    for i in range(len(tokens)):
        if i < len(labels2) and labels2[i] == "B-SEXO" and tokens[i].isdigit():
            hay_genero = any(
                labels2[j] in ("B-SEXO","I-SEXO") and tokens[j] in SEXO_TOKENS
                for j in range(max(0,i-5), i)
            )
            sig_anos = (i+1 < len(tokens) and
                        labels2[i+1] == "I-SEXO" and
                        tokens[i+1] in ("anos","an","a"))
            print(f"    Candidato tok={tokens[i]!r} pos={i}  hay_genero={hay_genero}  sig_anos={sig_anos}")
            if hay_genero and sig_anos:
                labels2[i] = "B-EDAD"
                labels2[i+1] = "O"
                print(f"    -> DISPARO: relabeled pos {i} a B-EDAD, pos {i+1} a O")

    b_edad_tras_r2 = [i for i,l in enumerate(labels2) if l == "B-EDAD"]
    b_dur_tras_r2  = [i for i,l in enumerate(labels2) if l == "B-DURACION"]
    print(f"  Tras Regla 2: B-EDAD={[(tokens[i],i) for i in b_edad_tras_r2]}")
    print(f"  Tras Regla 2: B-DURACION={[(tokens[i],i) for i in b_dur_tras_r2]}")

    # Paso 2: Regla 1
    print("\n  Regla 1 — duplicado B-EDAD:")
    if len(b_edad_tras_r2) > 1:
        for idx in b_edad_tras_r2[1:]:
            ini = max(0, idx - VENTANA)
            fin = min(len(tokens), idx + VENTANA + 1)
            ctx = set(tokens[ini:fin]) - {tokens[idx]}
            interseccion = ctx & DUR_CONTEXT
            print(f"    B-EDAD duplicado tok={tokens[idx]!r} pos={idx}")
            print(f"    Contexto ventana [{ini}:{fin}]: {list(tokens[ini:fin])}")
            print(f"    Interseccion con DUR_CONTEXT: {interseccion}")
            if interseccion:
                labels2[idx] = "B-DURACION"
                print(f"    -> DISPARO: relabeled a B-DURACION")
    else:
        print("    Solo 1 B-EDAD -> Regla 1 no dispara")

    b_dur_final = [(tokens[i], i) for i,l in enumerate(labels2) if l == "B-DURACION"]
    dur_result = next((int(tokens[i]) for i,l in enumerate(labels2)
                       if l=="B-DURACION" and tokens[i].isdigit()), None)
    print(f"\n  B-DURACION final: {b_dur_final}")
    print(f"  duracion extraida={dur_result}  esperado={dur_esp}  -> {'OK' if dur_result==dur_esp else 'FALLA'}")
