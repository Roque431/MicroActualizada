"""
Script de entrenamiento del modelo NER local para EpiDiagnostix-Mayab.
Ejecutar desde la raiz del proyecto:
    python entrenar_ner.py
"""
import os, sys, random, json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

import spacy
from spacy.training import Example
from spacy.util import minibatch
from spacy.scorer import Scorer

# ── Configuracion ────────────────────────────────────────────────────────────
SEED        = 42
N_EPOCHS    = 60
BATCH_SIZE  = 32
DROPOUT     = 0.35
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10
MAX_PAC     = 10   # early stopping patience

DATA_PATH  = Path("consultas_clinicas.csv")
MODEL_PATH = Path("models/ner_clinico_es")
Model_PATH_PARENT = Path("models")
Model_PATH_PARENT.mkdir(exist_ok=True)

ETIQUETAS = [
    "EDAD", "SEXO", "PESO_KG", "TALLA_CM",
    "PRESION_SIS", "PRESION_DIA",
    "GLUCOSA", "TEMPERATURA", "FREC_CARD",
    "DURACION", "CATEGORIA",
]

random.seed(SEED)
np.random.seed(SEED)
rng = random.Random(SEED)

# ── Funcion central de construccion de texto ─────────────────────────────────
def hacer_texto(*partes):
    texto_final = ""
    entidades   = []
    for texto, etiqueta in partes:
        if etiqueta and texto.strip():
            inicio = len(texto_final)
            fin    = inicio + len(texto)
            entidades.append((inicio, fin, etiqueta))
        texto_final += texto
    return texto_final, entidades

def fmt(v, d=1):
    """Formatea numero eliminando decimales innecesarios (.0)."""
    fv = float(v)
    if fv == int(fv):
        return str(int(fv))
    return str(round(fv, d))

def metros(cm):
    return str(round(float(cm) / 100, 2))

# ── Estilos de generacion ────────────────────────────────────────────────────
SEXO_M  = ["masculino", "hombre", "varon", "del sexo masculino"]
SEXO_F  = ["femenino", "mujer", "femenina", "del sexo femenino"]
SEXO_MC = ["senor", "nino"]
SEXO_FC = ["senora", "nina"]
INTRO_CAT = ["categoria: ", "categoria clinica: ", "impresion: ",
             "cuadro ", "se orienta a "]

def e1(r, rng_):   # SOAP formal
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    ic = rng_.choice(INTRO_CAT)
    return hacer_texto(
        ("Consulta medica. Paciente ", None), (sx,"SEXO"),
        (", ", None), (fmt(r.edad),"EDAD"), (" anios. Peso ", None),
        (fmt(r.peso_kg),"PESO_KG"), (" kg, talla ", None),
        (fmt(r.talla_cm),"TALLA_CM"), (" cm. Presion arterial ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (" mmHg. Glucosa ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (" mg/dL. Temperatura ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        ("C. FC ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (" lpm. Evolucion ", None), (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias. ", None), (ic, None), (r.categoria_sintoma,"CATEGORIA"), (".", None),
    )

def e2(r, rng_):   # Dictado rapido
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    return hacer_texto(
        ("Px ", None), (sx,"SEXO"), (", ", None),
        (fmt(r.edad),"EDAD"), (" anios. PA: ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Glucemia ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". T: ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (" grados. FC: ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (" lpm. Peso: ", None), (fmt(r.peso_kg),"PESO_KG"),
        (" kg. Talla: ", None), (fmt(r.talla_cm),"TALLA_CM"),
        (" cm. Evo: ", None), (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias.", None),
    )

def e3(r, rng_):   # Narrativo
    sx = rng_.choice(SEXO_MC if r.sexo=="M" else SEXO_FC)
    ic = rng_.choice(INTRO_CAT)
    return hacer_texto(
        ("Se atiende a ", None), (sx,"SEXO"), (" ", None),
        (fmt(r.edad),"EDAD"), (" anios. Peso ", None),
        (fmt(r.peso_kg),"PESO_KG"), (" kg, talla ", None),
        (fmt(r.talla_cm),"TALLA_CM"), (" cm. Presion ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" sobre ", None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Azucar en sangre ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". Temperatura ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (". Pulso ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (". Lleva ", None), (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias. ", None), (ic, None), (r.categoria_sintoma,"CATEGORIA"), (".", None),
    )

def e4(r, rng_):   # Coloquial
    prefix = "La " if r.sexo=="F" else "El "
    sx = rng_.choice(SEXO_FC if r.sexo=="F" else SEXO_MC)
    return hacer_texto(
        (prefix, None), (sx,"SEXO"), (" tiene como ", None),
        (fmt(r.edad),"EDAD"), (" anios. Presion ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" sobre ", None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Azucar ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". Temperatura ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (". Pulso ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (". Pesa ", None), (fmt(r.peso_kg),"PESO_KG"),
        (" kilos, mide ", None), (fmt(r.talla_cm),"TALLA_CM"),
        (" cm. Lleva ", None), (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias.", None),
    )

def e5(r, rng_):   # Talla en metros (critico)
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    ic = rng_.choice(INTRO_CAT)
    return hacer_texto(
        ("Paciente ", None), (sx,"SEXO"), (" de ", None),
        (fmt(r.edad),"EDAD"), (" anios. Peso ", None),
        (fmt(r.peso_kg),"PESO_KG"), (" kg, talla de ", None),
        (metros(r.talla_cm),"TALLA_CM"), (" m. Presion ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Glucosa en ayunas de ", None),
        (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (" mg. Temperatura ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (" grados. Frecuencia cardiaca de ", None),
        (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (" latidos. Evolucion de ", None),
        (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias. ", None), (ic, None), (r.categoria_sintoma,"CATEGORIA"), (".", None),
    )

def e6(r, rng_):   # Solo signos vitales
    return hacer_texto(
        ("Signos vitales: presion ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (", glucosa ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (" mg/dl, temperatura ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        ("C, FC ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (" lpm.", None),
    )

def e7(r, rng_):   # Orden alterado
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    return hacer_texto(
        ("PA ", None), (str(int(r.presion_sistolica)),"PRESION_SIS"),
        (" / ",None), (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Glucosa ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". T ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (". FC ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (". Px ", None), (sx,"SEXO"), (" de ", None),
        (fmt(r.edad),"EDAD"), (" anios. ", None),
        (fmt(r.peso_kg),"PESO_KG"), (" kg, ", None),
        (fmt(r.talla_cm),"TALLA_CM"), (" cm. ", None),
        (str(int(r.duracion_sintomas_dias)),"DURACION"), (" dias.", None),
    )

def e8(r, rng_):   # Solo datos basicos
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    return hacer_texto(
        ("Paciente ", None), (sx,"SEXO"), (", edad ", None),
        (fmt(r.edad),"EDAD"), (" anios. Peso: ", None),
        (fmt(r.peso_kg),"PESO_KG"), (" kilogramos. Estatura: ", None),
        (fmt(r.talla_cm),"TALLA_CM"), (" centimetros.", None),
    )

def e9(r, rng_):   # Variantes vocabulario
    sx = rng_.choice(SEXO_M if r.sexo=="M" else SEXO_F)
    return hacer_texto(
        ("Paciente ", None), (sx,"SEXO"), (", ", None),
        (fmt(r.edad),"EDAD"), (" anios. Tension arterial ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Glucemia capilar ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". Fiebre ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (" grados. Ritmo cardiaco ", None),
        (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (". Desde hace ", None), (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias. Cuadro ", None), (r.categoria_sintoma,"CATEGORIA"), (".", None),
    )

def e10(r, rng_):  # Solo categoria y duracion
    sx = rng_.choice(SEXO_MC if r.sexo=="M" else SEXO_FC)
    ic = rng_.choice(INTRO_CAT)
    return hacer_texto(
        ("Consulta de ", None), (sx,"SEXO"), (" de ", None),
        (fmt(r.edad),"EDAD"), (" anios. Lleva ", None),
        (str(int(r.duracion_sintomas_dias)),"DURACION"),
        (" dias. Presion ", None),
        (str(int(r.presion_sistolica)),"PRESION_SIS"), (" / ",None),
        (str(int(r.presion_diastolica)),"PRESION_DIA"),
        (". Glucosa ", None), (str(int(r.glucosa_mg_dl)),"GLUCOSA"),
        (". T ", None), (fmt(r.temperatura_c),"TEMPERATURA"),
        (". FC ", None), (str(int(r.frecuencia_cardiaca_bpm)),"FREC_CARD"),
        (". ", None), (ic, None), (r.categoria_sintoma,"CATEGORIA"), (".", None),
    )

ESTILOS = [e1,e2,e3,e4,e5,e6,e7,e8,e9,e10]

# ── Generar dataset ──────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df):,} filas")

print("Generando dataset...")
DATASET = []
for _, row in df.iterrows():
    for fn in ESTILOS:
        try:
            texto, ents = fn(row, rng)
            if texto and ents:
                DATASET.append((texto, {"entities": ents}))
        except Exception:
            pass

rng.shuffle(DATASET)
n = len(DATASET)
n_test  = int(n * TEST_SPLIT)
n_val   = int(n * VAL_SPLIT)
n_train = n - n_test - n_val

train_data = DATASET[:n_train]
val_data   = DATASET[n_train:n_train+n_val]
test_data  = DATASET[n_train+n_val:]

print(f"  Dataset: {n:,} ejemplos total")
print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

# ── Construir modelo ─────────────────────────────────────────────────────────
print("\nConstruyendo modelo spaCy...")
nlp = spacy.blank("es")
ner = nlp.add_pipe("ner", last=True)
for etiq in ETIQUETAS:
    ner.add_label(etiq)
print(f"  Etiquetas: {list(ner.labels)}")

def hacer_examples(nlp_, datos):
    exs = []
    for texto, ann in datos:
        doc = nlp_.make_doc(texto)
        try:
            exs.append(Example.from_dict(doc, ann))
        except Exception:
            pass
    return exs

def evaluar(nlp_, datos_eval):
    examples = hacer_examples(nlp_, datos_eval)
    if not examples:
        return 0.0, 0.0, 0.0
    scores = nlp_.evaluate(examples)
    return scores.get("ents_f",0), scores.get("ents_p",0), scores.get("ents_r",0)

# ── Training loop ────────────────────────────────────────────────────────────
print(f"\nIniciando entrenamiento: {N_EPOCHS} epocas, batch={BATCH_SIZE}, dropout={DROPOUT}")
optimizer = nlp.begin_training()
optimizer.learn_rate = 0.001

mejor_f1 = 0.0
paciencia = 0
historial = []

for epoch in range(N_EPOCHS):
    random.shuffle(train_data)
    losses = {}
    for batch in minibatch(train_data, size=BATCH_SIZE):
        examples = hacer_examples(nlp, batch)
        if examples:
            nlp.update(examples, sgd=optimizer, drop=DROPOUT, losses=losses)

    if (epoch+1) % 5 == 0 or epoch == 0:
        val_f1, val_p, val_r = evaluar(nlp, val_data)
        loss = losses.get("ner", 0)
        historial.append({"epoch":epoch+1,"loss":loss,"val_f1":val_f1,"val_p":val_p,"val_r":val_r})
        print(f"  Epoca {epoch+1:>3}/{N_EPOCHS} | Loss: {loss:8.1f} | Val F1: {val_f1:.4f}  P: {val_p:.4f}  R: {val_r:.4f}")

        if val_f1 > mejor_f1:
            mejor_f1 = val_f1
            paciencia = 0
            nlp.to_disk(MODEL_PATH)
        else:
            paciencia += 1
            if paciencia >= MAX_PAC:
                print(f"  Early stopping en epoca {epoch+1} — mejor F1: {mejor_f1:.4f}")
                break

print(f"\nEntrenamiento completo. Mejor Val F1: {mejor_f1:.4f}")

# ── Evaluacion final ─────────────────────────────────────────────────────────
print("\nEvaluando en TEST SET...")
nlp_final = spacy.load(MODEL_PATH)
test_examples = hacer_examples(nlp_final, test_data)
scores = nlp_final.evaluate(test_examples)

print(f"\n  === METRICAS EN TEST SET ===")
print(f"  F1  global: {scores['ents_f']:.4f}")
print(f"  Precision : {scores['ents_p']:.4f}")
print(f"  Recall    : {scores['ents_r']:.4f}")
print()
print(f"  {'Entidad':<18} {'F1':>8} {'P':>8} {'R':>8}")
print("  " + "-"*46)
epts = scores.get("ents_per_type", {})
for etiq in ETIQUETAS:
    if etiq in epts:
        s = epts[etiq]
        estado = "EXCELENTE" if s["f"]>=0.85 else "BUENO" if s["f"]>=0.70 else "MEJORAR"
        print(f"  {etiq:<18} {s['f']:>8.4f} {s['p']:>8.4f} {s['r']:>8.4f}  {estado}")
    else:
        print(f"  {etiq:<18} {'N/A':>8}")

# Guardar metricas
meta = {
    "dataset_size": n,
    "train": n_train, "val": n_val, "test": n_test,
    "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
    "val_f1_best": mejor_f1,
    "test_f1": scores["ents_f"],
    "test_precision": scores["ents_p"],
    "test_recall": scores["ents_r"],
    "per_entity": {e: epts.get(e, {}) for e in ETIQUETAS},
    "historial": historial,
}
with open("models/ner_clinico_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nModelo guardado en: {MODEL_PATH}")
print("Metricas guardadas en: models/ner_clinico_meta.json")
