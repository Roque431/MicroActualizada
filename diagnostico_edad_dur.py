import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, tensorflow as tf
tf.get_logger().setLevel("ERROR")

_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
def tok(t): return [m.group() for m in _RE.finditer(t.lower())]

with open("models/tflite_vocab_v2.json", encoding="utf-8") as f: VOCAB = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v:k for k,v in LABEL2ID.items()}
modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")

def predecir_completo(texto):
    tokens = tok(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens[:40]]
    x = np.array([ids + [0]*(40-len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    return tokens, [ID2LABEL[i] for i in pred]

CASOS = [
    ("Ciego 02 — edad real=33, duracion esperada=3",
     "Pues este paciente varon de 33 anos llego con 3 dias de evolucion. "
     "Peso 82 kg, talla 175 cm. Le tome la tension: 118/76 mmHg. "
     "Su azucar en sangre estaba 88. Temperatura 37.1. Frecuencia cardiaca 80.",
     33, 3),
    ("Ciego 07 — edad real=44, duracion esperada=9",
     "Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. "
     "Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78.",
     44, 9),
    ("Ciego 17 — edad real=36, duracion esperada=11",
     "Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. "
     "Le registre tension de 120/78 mmHg. Nivel de glucosa 88. "
     "Temperatura axilar 37.2. Pulso 76 por minuto.",
     36, 11),
]

for nombre, texto, edad_real, dur_esp in CASOS:
    tokens, labels = predecir_completo(texto)
    print(f"\n{'='*65}")
    print(f"  {nombre}")
    print(f"{'='*65}")
    print(f"  {'Token':<15} {'Label':<18}  Nota")
    print(f"  {'-'*55}")
    for t, l in zip(tokens, labels):
        nota = ""
        if t == str(edad_real):  nota = "<-- EDAD REAL"
        if t == str(dur_esp):    nota = "<-- DURACION ESPERADA"
        if l != "O":             linea = f"  {t:<15} {l:<18}  {nota}"
        else:                    linea = f"  {t:<15} {'O':<18}  {nota}"
        # Imprimir todos los que tienen entidad O son edad/dur relevantes
        if l != "O" or nota:
            print(linea)

    b_edad = [(t, i) for i,(t,l) in enumerate(zip(tokens,labels)) if l == "B-EDAD"]
    b_dur  = [(t, i) for i,(t,l) in enumerate(zip(tokens,labels)) if l == "B-DURACION"]
    print(f"\n  B-EDAD predichos:    {b_edad}")
    print(f"  B-DURACION predichos: {b_dur}")

    edad_ok  = any(t == str(edad_real) for t,_ in b_edad)
    dur_ok   = any(t == str(dur_esp)   for t,_ in b_dur)
    dur_wrong= any(t == str(dur_esp)   for t,_ in b_edad)  # dur en lugar de edad

    print(f"\n  Edad real ({edad_real}) etiquetada B-EDAD: {'SI' if edad_ok else 'NO'}")
    print(f"  Duracion ({dur_esp}) etiquetada B-DURACION: {'SI' if dur_ok else 'NO'}")
    print(f"  Duracion ({dur_esp}) etiquetada B-EDAD (error): {'SI' if dur_wrong else 'NO'}")

    if dur_wrong and not dur_ok:
        print(f"\n  PATRON: modelo etiqueta {dur_esp} como B-EDAD (error), NO como B-DURACION")
        if edad_ok:
            print(f"          Y TAMBIEN etiqueta {edad_real} como B-EDAD (correcto)")
            print(f"          => 'El modelo predice B-EDAD DOS VECES'")
        else:
            print(f"          Pero NO etiqueta {edad_real} como B-EDAD")
            print(f"          => 'El modelo predice B-EDAD UNA SOLA VEZ, en el numero incorrecto'")
    elif not dur_ok and not dur_wrong:
        print(f"\n  PATRON: {dur_esp} no recibe ninguna etiqueta de entidad")
