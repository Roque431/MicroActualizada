"""
FASE 1 — Generar dataset en formato BIO para el modelo TFLite on-device.

Reutiliza EXACTAMENTE los mismos estilos y funciones de entrenar_ner.py.
No los reescribe: los importa directamente.

Salida:
  models/tflite_vocab.json      — {token: id}  (para empaquetar en Flutter)
  models/tflite_labels.json     — {label: id}  (para decodificar en Dart)
  models/tflite_train.json      — [{tokens:[...], labels:[...], token_ids:[...], label_ids:[...]}, ...]
  models/tflite_val.json
  models/tflite_test.json
"""

import json
import re
import random
from collections import Counter
from pathlib import Path
import pandas as pd

# ── Importar TODO lo que ya existe en entrenar_ner.py ────────────────────────
# (funciones de generacion, estilos, fmt, metros, hacer_texto, ESTILOS, etc.)
import importlib.util, types, sys

spec = importlib.util.spec_from_file_location(
    "entrenar_ner",
    Path(__file__).parent / "entrenar_ner.py"
)
# Ejecutar el modulo hasta ANTES del bloque de entrenamiento (las funciones
# estan definidas en el top-level, las importamos sin correr el training loop)
_src = spec.loader.get_source("entrenar_ner")
# Extraer solo hasta el bloque de generacion del dataset (antes del training)
_cutoff = _src.index("# ── Construir modelo")
_mod_src = _src[:_cutoff]
_mod = types.ModuleType("entrenar_ner")
exec(compile(_mod_src, "entrenar_ner.py", "exec"), _mod.__dict__)

hacer_texto = _mod.hacer_texto
fmt         = _mod.fmt
metros      = _mod.metros
ESTILOS     = _mod.ESTILOS
DATASET     = _mod.DATASET      # ya generado por el exec
ETIQUETAS   = _mod.ETIQUETAS
SEED        = _mod.SEED

# ── Configuracion ─────────────────────────────────────────────────────────────
OUT_DIR  = Path("models")
VAL_SPLIT  = 0.10
TEST_SPLIT = 0.10
MIN_FREQ   = 2   # tokens con menos ocurrencias → <UNK>

# ── Tokenizador simple y determinista ─────────────────────────────────────────
# CRITICO: este mismo regex debe reimplementarse en Dart con RegExp identico.
# Captura:
#   \d+[.,]\d+  → numeros decimales (37.5, 1.72, 82.0)
#   \d+         → enteros           (45, 120, 95)
#   [a-z]+      → palabras en minusculas
# Todo lo demas (puntuacion, espacios, slashes) se descarta.
_TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')

def tokenizar(texto: str) -> list[tuple[str, int, int]]:
    """
    Devuelve lista de (token_str, char_inicio, char_fin).
    El texto se pasa a minusculas antes de buscar matches.
    """
    texto_lower = texto.lower()
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(texto_lower)]


# ── Conversion de spans de caracteres a etiquetas BIO ─────────────────────────
def spans_a_bio(tokens_con_pos: list, entidades: list) -> list[str]:
    """
    tokens_con_pos : [(token, char_start, char_end), ...]
    entidades      : [(char_start, char_end, etiqueta), ...]  ← formato spaCy
    Devuelve       : ["O", "B-EDAD", "O", "B-PRESION_SIS", ...]
    """
    # Mapa: posicion de caracter → etiqueta de la entidad que la cubre
    char_label = {}
    for ini, fin, etiq in entidades:
        for pos in range(ini, fin):
            char_label[pos] = etiq

    bio = []
    for token, t_ini, t_fin in tokens_con_pos:
        lbl = char_label.get(t_ini)
        if lbl is None:
            bio.append("O")
        else:
            # ¿Es el primer token de esta entidad?
            # Si el caracter anterior (t_ini - 1) no tiene la misma etiqueta → B-
            prev = char_label.get(t_ini - 1) if t_ini > 0 else None
            bio.append(f"B-{lbl}" if prev != lbl else f"I-{lbl}")
    return bio


# ── Construir ejemplos BIO ────────────────────────────────────────────────────
print("Construyendo ejemplos BIO...")
ejemplos_bio = []  # [{tokens:[str], labels:[str]}, ...]

for texto, ann in DATASET:
    tok_pos = tokenizar(texto)
    if not tok_pos:
        continue
    tokens = [t for t, _, _ in tok_pos]
    labels = spans_a_bio(tok_pos, ann["entities"])
    assert len(tokens) == len(labels)
    ejemplos_bio.append({"tokens": tokens, "labels": labels})

print(f"  {len(ejemplos_bio):,} ejemplos BIO generados")

# ── Vocabulario ───────────────────────────────────────────────────────────────
print("Construyendo vocabulario...")
freq = Counter(tok for ej in ejemplos_bio for tok in ej["tokens"])

VOCAB = {"<PAD>": 0, "<UNK>": 1}
for tok, cnt in sorted(freq.items()):
    if cnt >= MIN_FREQ:
        VOCAB[tok] = len(VOCAB)

print(f"  Tokens unicos totales : {len(freq):,}")
print(f"  Tokens con freq >= {MIN_FREQ}  : {len(VOCAB) - 2:,}  (+ <PAD> + <UNK>)")
print(f"  Vocabulario final     : {len(VOCAB):,}")

# Etiquetas BIO
todas_labels = sorted({lbl for ej in ejemplos_bio for lbl in ej["labels"]})
LABEL2ID = {lbl: i for i, lbl in enumerate(todas_labels)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}

print(f"  Etiquetas BIO         : {len(LABEL2ID)}")

# ── Convertir tokens a IDs ────────────────────────────────────────────────────
for ej in ejemplos_bio:
    ej["token_ids"] = [VOCAB.get(t, VOCAB["<UNK>"]) for t in ej["tokens"]]
    ej["label_ids"] = [LABEL2ID[l] for l in ej["labels"]]

# ── Split train / val / test ──────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(ejemplos_bio)
n = len(ejemplos_bio)
n_test = int(n * TEST_SPLIT)
n_val  = int(n * VAL_SPLIT)

test_bio  = ejemplos_bio[:n_test]
val_bio   = ejemplos_bio[n_test:n_test + n_val]
train_bio = ejemplos_bio[n_test + n_val:]

print(f"\nSplit: train={len(train_bio):,}  val={len(val_bio):,}  test={len(test_bio):,}")

# ── Guardar artefactos ────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)

with open(OUT_DIR / "tflite_vocab.json",  "w", encoding="utf-8") as f:
    json.dump(VOCAB, f, ensure_ascii=False, indent=2)

with open(OUT_DIR / "tflite_labels.json", "w", encoding="utf-8") as f:
    json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)

for nombre, split in [("train", train_bio), ("val", val_bio), ("test", test_bio)]:
    with open(OUT_DIR / f"tflite_{nombre}.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False)
    sz = (OUT_DIR / f"tflite_{nombre}.json").stat().st_size / 1024
    print(f"  tflite_{nombre}.json  — {len(split):,} ejemplos  ({sz:.0f} KB)")

# ── Estadisticas ──────────────────────────────────────────────────────────────
longitudes = [len(ej["tokens"]) for ej in ejemplos_bio]
label_cnt  = Counter(lbl for ej in ejemplos_bio for lbl in ej["labels"])

print("\n" + "=" * 62)
print("  ESTADISTICAS — Dataset BIO para TFLite")
print("=" * 62)
print(f"  Ejemplos totales      : {n:,}")
print(f"  Vocabulario           : {len(VOCAB):,} tokens  (PAD+UNK incluidos)")
print(f"  Etiquetas BIO         : {len(LABEL2ID)}")
print()
print(f"  Longitud de secuencia (en tokens):")
print(f"    Promedio   : {sum(longitudes)/len(longitudes):.1f}")
print(f"    Maxima     : {max(longitudes)}")
print(f"    Minima     : {min(longitudes)}")
print()
print(f"  Distribucion de etiquetas BIO:")
total_labels = sum(label_cnt.values())
# Mostrar O primero, luego B- y I- agrupadas por entidad
print(f"    {'O':<20} {label_cnt['O']:>8,}  ({label_cnt['O']/total_labels*100:.1f}%)")
print()
for etiq in ETIQUETAS:
    b = label_cnt.get(f"B-{etiq}", 0)
    i = label_cnt.get(f"I-{etiq}", 0)
    if b + i > 0:
        print(f"    B-{etiq:<17} {b:>8,}  |  I-{etiq:<17} {i:>8,}")
print()
print(f"  Ratio O vs entidades  : {label_cnt['O']:,} O  /  {total_labels - label_cnt['O']:,} entidad")
print()
print("  Archivos guardados en models/:")
print("    tflite_vocab.json   — vocabulario token→id  (para empaquetar en Flutter)")
print("    tflite_labels.json  — etiquetas label→id    (para decodificar en Dart)")
print("    tflite_train/val/test.json — dataset BIO con tokens e IDs")
print("=" * 62)

# ── Muestra de 3 ejemplos para inspeccion visual ─────────────────────────────
print("\n  EJEMPLOS (3 aleatorios para validar BIO):")
for ej in random.sample(ejemplos_bio, 3):
    print()
    for tok, lbl in zip(ej["tokens"], ej["labels"]):
        if lbl != "O":
            print(f"    [{lbl:>18}]  {tok}")
        else:
            print(f"    {'O':>18}   {tok}")
