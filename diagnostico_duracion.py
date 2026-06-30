"""
Diagnóstico: ¿los errores de duracion_sintomas_dias en el set ciego
contienen "días"/"años" con acento (que fragmenta el tokenizador)?
"""
import sys, io, re, json, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

_TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')
def tokenizar(texto):
    return [m.group() for m in _TOKEN_RE.finditer(texto.lower())]

with open("models/tflite_vocab_v2.json", encoding="utf-8") as f:
    VOCAB = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f:
    LABEL2ID = json.load(f)
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")
MAX_LEN = 40

def inferir(texto):
    tokens = tokenizar(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens[:MAX_LEN]]
    x = np.array([ids + [0]*(MAX_LEN-len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    labels = [ID2LABEL[i] for i in pred]
    # Extraer duracion
    for tok, lbl in zip(tokens, labels):
        if lbl == "B-DURACION":
            try: return int(tok), tokens, labels
            except: pass
    return None, tokens, labels

CASOS_CIEGOS = [
    ("Ciego 01",
     "Tension arterial 128/82. Calentura de 37.9 grados. Glucemia 91 mg/dl. "
     "FC 76 lpm. Senora de 52 anos. Peso 68 kilos, estatura 160 cm. Desde hace 5 dias.",
     {"duracion_sintomas_dias": 5}),
    ("Ciego 02",
     "Pues este paciente varon de 33 anos llego con 3 dias de evolucion. "
     "Peso 82 kg, talla 175 cm. Le tome la tension: 118/76 mmHg. "
     "Su azucar en sangre estaba 88. Temperatura 37.1. Frecuencia cardiaca 80.",
     {"duracion_sintomas_dias": 3}),
    ("Ciego 03",
     "Paciente femenino de 26 anos. Signos: TA 110/70, T 36.5, FC 72, Glx 85 mg/dl. "
     "Peso 55.5 kg, talla 1.58 m. Tiempo de evolucion: 7 dias.",
     {"duracion_sintomas_dias": 7}),
    ("Ciego 04",
     "Se atiende a caballero de 71 anos. Calentura de 38.1 grados. "
     "Presion 162/98 mmHg. Pulso 94 latidos. Nivel de glucosa 145. "
     "Peso 79 kg, mide 163 cm. Lleva una semana enfermo.",
     {"duracion_sintomas_dias": 7}),
    ("Ciego 05",
     "Doctor, le informo sobre el nino de 7 anos. Masa corporal 22 kg, altura 118 cm. "
     "Temperatura axilar 39.2 grados. Ritmo cardiaco 115 bpm. PA 90/55. Inicio hace 2 dias.",
     {"duracion_sintomas_dias": 2}),
    ("Ciego 06",
     "Paciente masculino, 54 anos, 80.5 kilogramos, 172 centimetros. "
     "Tension: 140/88. Glicemia en ayunas: 118. Temperatura: 36.9. Pulso: 84. Cuadro de 4 dias.",
     {"duracion_sintomas_dias": 4}),
    ("Ciego 07",
     "Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. "
     "Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78.",
     {"duracion_sintomas_dias": 9}),
    ("Ciego 08",
     "Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, "
     "FC 68, glucosa 92. Cuadro agudo de 1 dia.",
     {"duracion_sintomas_dias": 1}),
    ("Ciego 09",
     "Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. "
     "La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88.",
     {"duracion_sintomas_dias": 10}),
    ("Ciego 10",
     "Escolar femenino de 11 anos. Talla 140 cm, peso 36 kg. "
     "Temperatura de 38.6 grados. Pulso 98. Presion 95/60. Glucemia 88 mg/dl. 3 dias.",
     {"duracion_sintomas_dias": 3}),
    ("Ciego 11",
     "Hombre de 47 anos. Viene por cuadro de 8 dias. Peso actual: 91 kg. Estatura: 178 cm. "
     "Signos: tension 138/90, azucar 126 mg/dl, T 36.6C, latidos 82 por minuto.",
     {"duracion_sintomas_dias": 8}),
    ("Ciego 12",
     "Me llego este paciente femenino de 29 anos, con fiebre de 39.5 y llevaba 4 dias asi. "
     "La presion la tenia en 105/68. El pulso en 102.",
     {"duracion_sintomas_dias": 4}),
    ("Ciego 13",
     "Datos del paciente: edad 58 anos, sexo masculino. Medidas: 74 kg / 166 cm. "
     "Signos vitales: tension 148/92 mmHg, glucemia 210 mg/dl, temperatura 37.0 grados, "
     "FC 88 lpm. Evolucion: 6 dias.",
     {"duracion_sintomas_dias": 6}),
    ("Ciego 14",
     "Adulto mayor de sexo femenino, 78 anos. Refiere inicio hace 2 dias. "
     "Peso 52 kg. Talla 149 cm. La calentura: 37.8. Azucar en ayunas: 135. "
     "Tension: 168/102. Latidos: 92.",
     {"duracion_sintomas_dias": 2}),
    ("Ciego 15",
     "Paciente de 40 anos, sexo masculino. Sin datos de peso y talla. "
     "Tension arterial: 122/80. Temperatura: 38.0C. Glucosa: 99. "
     "Frecuencia cardiaca: 90. Evolucion de 3 dias.",
     {"duracion_sintomas_dias": 3}),
    ("Ciego 16",
     "Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. "
     "TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. "
     "Peso 67.5 kg. Estatura 1.74 m.",
     {"duracion_sintomas_dias": 5}),
    ("Ciego 17",
     "Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. "
     "Le registre tension de 120/78 mmHg. Nivel de glucosa 88. "
     "Temperatura axilar 37.2. Pulso 76 por minuto.",
     {"duracion_sintomas_dias": 11}),
    ("Ciego 18",
     "Nino de 5 anos, masculino. Fiebre 38.4C de 3 dias de evolucion. "
     "Peso 18 kg, mide 108 cm. Tension 88/58. FC 108. Glucosa 85.",
     {"duracion_sintomas_dias": 3}),
    ("Ciego 19",
     "Consulto paciente femenino de 62 anos con 14 dias de cuadro. "
     "Mide 1.53 m, pesa 71 kilos. Tension 158/96. Calentura 37.6. "
     "Azucar en sangre 142 mg/dl. Pulso 86 lpm.",
     {"duracion_sintomas_dias": 14}),
    ("Ciego 20",
     "Masculino, 15 anos. Talla 165 cm, peso 55 kg. Temperatura de 39.0 grados. "
     "Pulso 105. Tension arterial 100/65. Glucosa 90. Inicio hace 1 dia.",
     {"duracion_sintomas_dias": 1}),
]

print("="*75)
print("DIAGNOSTICO: errores de duracion_sintomas_dias en set ciego")
print("="*75)
print(f"{'Caso':<10} {'Esp':>5} {'Pred':>5} {'OK':>4}  {'Texto de duracion en frase'}")
print("-"*75)

errores = []
for nombre, texto, esp in CASOS_CIEGOS:
    dur_esp = esp["duracion_sintomas_dias"]
    pred_dur, tokens, labels = inferir(texto)

    # Encontrar los tokens que el modelo etiqueto como DURACION
    dur_tokens = [(tok, lbl) for tok, lbl in zip(tokens, labels) if "DURACION" in lbl]

    ok = (pred_dur is not None and abs(pred_dur - dur_esp) <= 1)

    # Analizar el texto: ¿tiene "dias"/"dias" con/sin acento? ¿"semana"? ¿numero de dias?
    texto_lower = texto.lower()
    tiene_dias_acento   = "días" in texto_lower  # días
    tiene_dias_sin      = "dias" in texto_lower
    tiene_semana        = "semana" in texto_lower
    tiene_dia_singular  = "1 dia" in texto_lower or "un dia" in texto_lower or "1 día" in texto_lower

    contexto = ""
    if tiene_dias_acento:   contexto += "[DIAS CON ACENTO] "
    if tiene_dias_sin:      contexto += "[dias sin acento] "
    if tiene_semana:        contexto += "[SEMANA - sin numero!] "
    if tiene_dia_singular:  contexto += "[dia singular] "

    estado = "OK" if ok else "FALLA"
    print(f"{nombre:<10} {dur_esp:>5} {str(pred_dur):>5} {estado:>4}  {contexto}")

    if not ok:
        # Mostrar tokens alrededor de la duracion esperada
        dur_str = str(dur_esp)
        idx = next((i for i,t in enumerate(tokens) if t == dur_str), None)
        ventana = tokens[max(0,(idx or 0)-3):(idx or 0)+4] if idx is not None else []
        print(f"           Tokens near {dur_esp}: {ventana}")
        print(f"           Labels:                {labels[max(0,(idx or 0)-3):(idx or 0)+4] if idx is not None else '(numero no encontrado en tokens)'}")
        print(f"           DURACION labels:        {dur_tokens}")
        errores.append(nombre)

print()
print("="*75)
print(f"Errores de duracion: {len(errores)}/20")
print(f"Casos fallidos: {errores}")
print()

# Respuesta directa a la pregunta del usuario
print("RESPUESTA A LA PREGUNTA:")
acento_en_error = 0
semana_en_error = 0
sin_acento_en_error = 0
for nombre, texto, esp in CASOS_CIEGOS:
    if nombre in errores:
        if "días" in texto.lower() or "años" in texto.lower():
            acento_en_error += 1
            print(f"  {nombre}: SI tiene acento -> tokenizador fragmenta")
        elif "semana" in texto.lower():
            semana_en_error += 1
            print(f"  {nombre}: 'semana' en vez de numero de dias -> diferente problema")
        else:
            sin_acento_en_error += 1
            print(f"  {nombre}: 'dias' sin acento -> el tokenizador NO es el culpable aqui")
