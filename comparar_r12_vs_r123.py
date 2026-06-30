"""
Comparación campo a campo: R1+R2 vs R1+R2+R3 sobre los 189 campos de SET3.
Lista TODOS los campos que cambiaron entre las dos configuraciones.
"""
import sys, io, re, json, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, tensorflow as tf
tf.get_logger().setLevel("ERROR")

with open("models/tflite_vocab_v2.json",  encoding="utf-8") as f: VOCAB    = json.load(f)
with open("models/tflite_labels_v2.json", encoding="utf-8") as f: LABEL2ID = json.load(f)
ID2LABEL = {v:k for k,v in LABEL2ID.items()}
modelo = tf.keras.models.load_model("models/ner_tflite_v2.keras")

_RE = re.compile(r"\d+[.,]\d+|\d+|[a-z]+")
def tokenizar(t): return [m.group() for m in _RE.finditer(t.lower())][:40]
def norm(s): return unicodedata.normalize("NFD",str(s)).encode("ascii","ignore").decode().lower().strip()

def raw_labels(texto):
    tokens = tokenizar(texto)
    ids = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
    x = np.array([ids + [0]*(40-len(ids))], dtype=np.int32)
    pred = modelo.predict(x, verbose=0)[0].argmax(axis=-1)[:len(tokens)]
    return tokens, [ID2LABEL[i] for i in pred]

SEXO_TOKENS = {"mujer","hombre","varon","masculino","femenino","femenina",
               "senora","senor","nina","nino","dama","caballero","muchacha","muchacho"}
DUR_CONTEXT = {"dias","dia","evolucion","cuadro","lleva","desde","hace"}
SEXO_MAP = {"masculino":"M","hombre":"M","varon":"M","nino":"M","senor":"M","caballero":"M",
            "mujer":"F","femenino":"F","femenina":"F","nina":"F","senora":"F","dama":"F"}
CAT_MAP  = {"gastrointestinal":"Gastrointestinal","respiratorio":"Respiratorio",
            "hipertension":"Hipertension","diabetes":"Diabetes","dengue":"Infeccioso/Vectorial",
            "vacunacion":"Vacunacion","nutricion":"Nutricion"}

def to_int(v):
    try: return int(float(str(v).replace(",",".")))
    except: return None
def to_float(v):
    try: return round(float(str(v).replace(",",".")),1)
    except: return None

def aplicar_reglas_1_2(tokens, labels):
    labels = list(labels)
    for i in range(len(labels)):
        if i < len(tokens) and labels[i]=="B-SEXO" and tokens[i].isdigit():
            hay_genero = any(labels[j] in ("B-SEXO","I-SEXO") and tokens[j] in SEXO_TOKENS
                             for j in range(max(0,i-5),i))
            sig_anos = i+1<len(tokens) and labels[i+1]=="I-SEXO" and tokens[i+1] in ("anos","an","a")
            if hay_genero and sig_anos:
                labels[i]="B-EDAD"; labels[i+1]="O"
    edad_idx = [i for i,l in enumerate(labels) if l=="B-EDAD"]
    if len(edad_idx) > 1:
        for idx in edad_idx[1:]:
            ini=max(0,idx-4); fin=min(len(tokens),idx+5)
            if set(tokens[ini:fin])-{tokens[idx]} & DUR_CONTEXT:
                labels[idx]="B-DURACION"
    return labels

def extraer_R12(tokens, labels):
    """Extractor original (R1+R2): toma primer B-SEXO aunque sea None."""
    res = {}
    for tok, lbl in zip(tokens, labels):
        if lbl=="O" or not lbl.startswith("B-"): continue
        base = lbl[2:]
        if base=="EDAD" and "edad" not in res:
            v=to_int(tok)
            if v and 0<v<=120: res["edad"]=v
        elif base=="SEXO" and "sexo" not in res:
            res["sexo"] = SEXO_MAP.get(norm(tok))   # puede quedar None
        elif base=="PESO_KG"   and "peso_kg"   not in res: res["peso_kg"]  = to_float(tok)
        elif base=="TALLA_CM"  and "talla_cm"  not in res:
            v=to_float(tok); res["talla_cm"]=round(v*100,1) if v and v<3 else v
        elif base=="PRESION_SIS"  and "presion_sistolica"  not in res: res["presion_sistolica"] =to_int(tok)
        elif base=="PRESION_DIA"  and "presion_diastolica" not in res: res["presion_diastolica"]=to_int(tok)
        elif base=="GLUCOSA"   and "glucosa_mg_dl"       not in res: res["glucosa_mg_dl"]     =to_int(tok)
        elif base=="TEMPERATURA" and "temperatura_c"     not in res: res["temperatura_c"]    =to_float(tok)
        elif base=="FREC_CARD"   and "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"]=to_int(tok)
        elif base=="DURACION"  and "duracion_sintomas_dias" not in res:
            v=to_int(tok)
            if v is not None: res["duracion_sintomas_dias"]=v
        elif base=="CATEGORIA" and "categoria_sintoma" not in res:
            res["categoria_sintoma"]=CAT_MAP.get(norm(tok),tok)
    return res

def extraer_R123(tokens, labels):
    """Extractor con R3: itera todos los SEXO, toma primero válido."""
    res = {}
    # Regla 3: SEXO
    for tok, lbl in zip(tokens, labels):
        if "SEXO" in lbl:
            v = SEXO_MAP.get(norm(tok))
            if v is not None:
                res["sexo"] = v
                break
    # Resto igual
    for tok, lbl in zip(tokens, labels):
        if lbl=="O" or not lbl.startswith("B-"): continue
        base = lbl[2:]
        if base=="EDAD" and "edad" not in res:
            v=to_int(tok)
            if v and 0<v<=120: res["edad"]=v
        elif base=="PESO_KG"   and "peso_kg"   not in res: res["peso_kg"]  = to_float(tok)
        elif base=="TALLA_CM"  and "talla_cm"  not in res:
            v=to_float(tok); res["talla_cm"]=round(v*100,1) if v and v<3 else v
        elif base=="PRESION_SIS"  and "presion_sistolica"  not in res: res["presion_sistolica"] =to_int(tok)
        elif base=="PRESION_DIA"  and "presion_diastolica" not in res: res["presion_diastolica"]=to_int(tok)
        elif base=="GLUCOSA"   and "glucosa_mg_dl"       not in res: res["glucosa_mg_dl"]     =to_int(tok)
        elif base=="TEMPERATURA" and "temperatura_c"     not in res: res["temperatura_c"]    =to_float(tok)
        elif base=="FREC_CARD"   and "frecuencia_cardiaca_bpm" not in res: res["frecuencia_cardiaca_bpm"]=to_int(tok)
        elif base=="DURACION"  and "duracion_sintomas_dias" not in res:
            v=to_int(tok)
            if v is not None: res["duracion_sintomas_dias"]=v
        elif base=="CATEGORIA" and "categoria_sintoma" not in res:
            res["categoria_sintoma"]=CAT_MAP.get(norm(tok),tok)
    return res

TOL = {"edad":1,"peso_kg":1,"talla_cm":2,"presion_sistolica":1,"presion_diastolica":1,
       "glucosa_mg_dl":1,"temperatura_c":0.2,"frecuencia_cardiaca_bpm":1,"duracion_sintomas_dias":1}
def ok(pred, esp, campo):
    if pred is None: return False
    tol=TOL.get(campo)
    if tol: return abs(float(pred)-float(esp))<=tol
    return norm(str(pred))==norm(str(esp))

SET3 = [
    ("C01","Tension arterial 128/82. Calentura de 37.9 grados. Glucemia 91 mg/dl. FC 76 lpm. Senora de 52 anos. Peso 68 kilos, estatura 160 cm. Desde hace 5 dias.",{"sexo":"F","edad":52,"presion_sistolica":128,"presion_diastolica":82,"temperatura_c":37.9,"glucosa_mg_dl":91,"frecuencia_cardiaca_bpm":76,"peso_kg":68,"talla_cm":160,"duracion_sintomas_dias":5}),
    ("C02","Pues este paciente varon de 33 anos llego con 3 dias de evolucion. Peso 82 kg, talla 175 cm. Le tome la tension: 118/76 mmHg. Su azucar en sangre estaba 88. Temperatura 37.1. Frecuencia cardiaca 80.",{"sexo":"M","edad":33,"duracion_sintomas_dias":3,"peso_kg":82,"talla_cm":175,"presion_sistolica":118,"presion_diastolica":76,"glucosa_mg_dl":88,"temperatura_c":37.1,"frecuencia_cardiaca_bpm":80}),
    ("C03","Paciente femenino de 26 anos. Signos: TA 110/70, T 36.5, FC 72, Glx 85 mg/dl. Peso 55.5 kg, talla 1.58 m. Tiempo de evolucion: 7 dias.",{"sexo":"F","edad":26,"presion_sistolica":110,"presion_diastolica":70,"temperatura_c":36.5,"frecuencia_cardiaca_bpm":72,"glucosa_mg_dl":85,"peso_kg":55.5,"talla_cm":158,"duracion_sintomas_dias":7}),
    ("C04","Se atiende a caballero de 71 anos. Calentura de 38.1 grados. Presion 162/98 mmHg. Pulso 94 latidos. Nivel de glucosa 145. Peso 79 kg, mide 163 cm. Lleva una semana enfermo.",{"sexo":"M","edad":71,"temperatura_c":38.1,"presion_sistolica":162,"presion_diastolica":98,"frecuencia_cardiaca_bpm":94,"glucosa_mg_dl":145,"peso_kg":79,"talla_cm":163,"duracion_sintomas_dias":7}),
    ("C05","Doctor, le informo sobre el nino de 7 anos. Masa corporal 22 kg, altura 118 cm. Temperatura axilar 39.2 grados. Ritmo cardiaco 115 bpm. PA 90/55. Inicio hace 2 dias.",{"sexo":"M","edad":7,"peso_kg":22,"talla_cm":118,"temperatura_c":39.2,"frecuencia_cardiaca_bpm":115,"presion_sistolica":90,"presion_diastolica":55,"duracion_sintomas_dias":2}),
    ("C06","Paciente masculino, 54 anos, 80.5 kilogramos, 172 centimetros. Tension: 140/88. Glicemia en ayunas: 118. Temperatura: 36.9. Pulso: 84. Cuadro de 4 dias.",{"sexo":"M","edad":54,"peso_kg":80.5,"talla_cm":172,"presion_sistolica":140,"presion_diastolica":88,"glucosa_mg_dl":118,"temperatura_c":36.9,"frecuencia_cardiaca_bpm":84,"duracion_sintomas_dias":4}),
    ("C07","Mujer de 44 anos que refiere 9 dias de evolucion. Pesa 61 kg, mide 155 cm. Temperatura: 37.3C. Azucar: 97 mg/dl. Presion: 125/80. FC: 78.",{"sexo":"F","edad":44,"duracion_sintomas_dias":9,"peso_kg":61,"talla_cm":155,"temperatura_c":37.3,"glucosa_mg_dl":97,"presion_sistolica":125,"presion_diastolica":80,"frecuencia_cardiaca_bpm":78}),
    ("C08","Masculino de 18 anos. Ingresa con PA 108/65, temperatura 36.7, FC 68, glucosa 92. Cuadro agudo de 1 dia.",{"sexo":"M","edad":18,"presion_sistolica":108,"presion_diastolica":65,"temperatura_c":36.7,"frecuencia_cardiaca_bpm":68,"glucosa_mg_dl":92,"duracion_sintomas_dias":1}),
    ("C09","Entonces la senora esta, tiene 65 anos, lleva como 10 dias enferma. La tension le salio alta, en 175 sobre 105. La temperatura era de 37.4 y el pulso 88.",{"sexo":"F","edad":65,"duracion_sintomas_dias":10,"presion_sistolica":175,"presion_diastolica":105,"temperatura_c":37.4,"frecuencia_cardiaca_bpm":88}),
    ("C10","Escolar femenino de 11 anos. Talla 140 cm, peso 36 kg. Temperatura de 38.6 grados. Pulso 98. Presion 95/60. Glucemia 88 mg/dl. 3 dias.",{"sexo":"F","edad":11,"talla_cm":140,"peso_kg":36,"temperatura_c":38.6,"frecuencia_cardiaca_bpm":98,"presion_sistolica":95,"presion_diastolica":60,"glucosa_mg_dl":88,"duracion_sintomas_dias":3}),
    ("C11","Hombre de 47 anos. Viene por cuadro de 8 dias. Peso actual: 91 kg. Estatura: 178 cm. Signos: tension 138/90, azucar 126 mg/dl, T 36.6C, latidos 82 por minuto.",{"sexo":"M","edad":47,"duracion_sintomas_dias":8,"peso_kg":91,"talla_cm":178,"presion_sistolica":138,"presion_diastolica":90,"glucosa_mg_dl":126,"temperatura_c":36.6,"frecuencia_cardiaca_bpm":82}),
    ("C12","Me llego este paciente femenino de 29 anos, con fiebre de 39.5 y llevaba 4 dias asi. La presion la tenia en 105/68. El pulso en 102.",{"sexo":"F","edad":29,"temperatura_c":39.5,"duracion_sintomas_dias":4,"presion_sistolica":105,"presion_diastolica":68,"frecuencia_cardiaca_bpm":102}),
    ("C13","Datos del paciente: edad 58 anos, sexo masculino. Medidas: 74 kg / 166 cm. Signos vitales: tension 148/92 mmHg, glucemia 210 mg/dl, temperatura 37.0 grados, FC 88 lpm. Evolucion: 6 dias.",{"edad":58,"sexo":"M","peso_kg":74,"talla_cm":166,"presion_sistolica":148,"presion_diastolica":92,"glucosa_mg_dl":210,"temperatura_c":37.0,"frecuencia_cardiaca_bpm":88,"duracion_sintomas_dias":6}),
    ("C14","Adulto mayor de sexo femenino, 78 anos. Refiere inicio hace 2 dias. Peso 52 kg. Talla 149 cm. La calentura: 37.8. Azucar en ayunas: 135. Tension: 168/102. Latidos: 92.",{"sexo":"F","edad":78,"duracion_sintomas_dias":2,"peso_kg":52,"talla_cm":149,"temperatura_c":37.8,"glucosa_mg_dl":135,"presion_sistolica":168,"presion_diastolica":102,"frecuencia_cardiaca_bpm":92}),
    ("C15","Paciente de 40 anos, sexo masculino. Sin datos de peso y talla. Tension arterial: 122/80. Temperatura: 38.0C. Glucosa: 99. Frecuencia cardiaca: 90. Evolucion de 3 dias.",{"edad":40,"sexo":"M","presion_sistolica":122,"presion_diastolica":80,"temperatura_c":38.0,"glucosa_mg_dl":99,"frecuencia_cardiaca_bpm":90,"duracion_sintomas_dias":3}),
    ("C16","Atiende medico a paciente masculino de 25 anos. Acude por cuadro de 5 dias. TA 115/72. Temperatura 36.4. FC 70. Glucemia capilar 90 mg/dl. Peso 67.5 kg. Estatura 1.74 m.",{"sexo":"M","edad":25,"duracion_sintomas_dias":5,"presion_sistolica":115,"presion_diastolica":72,"temperatura_c":36.4,"frecuencia_cardiaca_bpm":70,"glucosa_mg_dl":90,"peso_kg":67.5,"talla_cm":174}),
    ("C17","Esta dama de 36 anos dice llevar 11 dias con molestias. Peso 58 kg, altura 162 cm. Le registre tension de 120/78 mmHg. Nivel de glucosa 88. Temperatura axilar 37.2. Pulso 76 por minuto.",{"sexo":"F","edad":36,"duracion_sintomas_dias":11,"peso_kg":58,"talla_cm":162,"presion_sistolica":120,"presion_diastolica":78,"glucosa_mg_dl":88,"temperatura_c":37.2,"frecuencia_cardiaca_bpm":76}),
    ("C18","Nino de 5 anos, masculino. Fiebre 38.4C de 3 dias de evolucion. Peso 18 kg, mide 108 cm. Tension 88/58. FC 108. Glucosa 85.",{"sexo":"M","edad":5,"temperatura_c":38.4,"duracion_sintomas_dias":3,"peso_kg":18,"talla_cm":108,"presion_sistolica":88,"presion_diastolica":58,"frecuencia_cardiaca_bpm":108,"glucosa_mg_dl":85}),
    ("C19","Consulto paciente femenino de 62 anos con 14 dias de cuadro. Mide 1.53 m, pesa 71 kilos. Tension 158/96. Calentura 37.6. Azucar en sangre 142 mg/dl. Pulso 86 lpm.",{"sexo":"F","edad":62,"duracion_sintomas_dias":14,"talla_cm":153,"peso_kg":71,"presion_sistolica":158,"presion_diastolica":96,"temperatura_c":37.6,"glucosa_mg_dl":142,"frecuencia_cardiaca_bpm":86}),
    ("C20","Masculino, 15 anos. Talla 165 cm, peso 55 kg. Temperatura de 39.0 grados. Pulso 105. Tension arterial 100/65. Glucosa 90. Inicio hace 1 dia.",{"sexo":"M","edad":15,"talla_cm":165,"peso_kg":55,"temperatura_c":39.0,"frecuencia_cardiaca_bpm":105,"presion_sistolica":100,"presion_diastolica":65,"glucosa_mg_dl":90,"duracion_sintomas_dias":1}),
]

print("Comparacion campo a campo: R1+R2 vs R1+R2+R3 (SET3, 189 campos)\n")
print(f"{'Caso':<6}  {'Campo':<28}  {'R1+R2':>8}  {'R1+R2+R3':>10}  {'Esp':>6}  {'Cambio'}")
print("-"*75)

total_ok_12 = total_ok_123 = 0
cambios = []

for nombre, texto, esp in SET3:
    tokens, labels_raw = raw_labels(texto)
    labels_pp = aplicar_reglas_1_2(tokens, labels_raw)
    res_12  = extraer_R12(tokens, labels_pp)
    res_123 = extraer_R123(tokens, labels_pp)

    for campo, val_esp in esp.items():
        v12  = res_12.get(campo)
        v123 = res_123.get(campo)
        ok12  = ok(v12,  val_esp, campo)
        ok123 = ok(v123, val_esp, campo)
        total_ok_12  += ok12
        total_ok_123 += ok123

        if v12 != v123:  # cualquier cambio, sea cual sea la dirección
            tipo = ("INC->COR" if not ok12 and ok123 else
                    "COR->INC" if ok12 and not ok123 else
                    "INC->INC")
            cambios.append({
                "caso": nombre, "campo": campo,
                "esp": val_esp, "r12": v12, "r123": v123,
                "ok12": ok12, "ok123": ok123, "tipo": tipo
            })
            print(f"{nombre:<6}  {campo:<28}  {str(v12):>8}  {str(v123):>10}  {str(val_esp):>6}  {tipo}")

print("\n" + "="*75)
print(f"Total campos evaluados : 189")
print(f"Correctos con R1+R2    : {total_ok_12}/189  ({total_ok_12/189*100:.1f}%)")
print(f"Correctos con R1+R2+R3 : {total_ok_123}/189  ({total_ok_123/189*100:.1f}%)")
print(f"Delta                  : {total_ok_123 - total_ok_12:+d}")
print(f"\nCampos que cambiaron   : {len(cambios)}")
for tipo in ["INC->COR", "COR->INC", "INC->INC"]:
    n = sum(1 for c in cambios if c["tipo"]==tipo)
    if n: print(f"  {tipo}: {n}")
