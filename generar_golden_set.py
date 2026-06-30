"""
Genera el golden set de paridad tokenizador Python → Dart.
35 frases representativas del dataset de entrenamiento:
  - ASCII puro (como en training data)
  - Con acentos (como llega desde STT real)
  - Números con coma y punto decimal
  - Slashes (presión "X/Y"), abreviaciones médicas
  - Mayúsculas/minúsculas mezcladas
  - Frases incompletas / edge cases

Salida: models/tokenizer_golden_set.json
  {"frases": [{"original": ..., "tokens": [...], "ids": [...]}]}
"""
import re, json
from pathlib import Path

_TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')

def tokenizar(texto):
    """Tokenizador exacto usado en el entrenamiento — NO modificar."""
    return [m.group() for m in _TOKEN_RE.finditer(texto.lower())]

VOCAB_PATH = Path("models/tflite_vocab_v2.json")
with open(VOCAB_PATH, encoding="utf-8") as f:
    VOCAB = json.load(f)

def ids(tokens):
    unk = VOCAB["<UNK>"]
    return [VOCAB.get(t, unk) for t in tokens]

# ── 35 frases de prueba ───────────────────────────────────────────────────────
FRASES = [
    # ── Grupo 1: ASCII puro (idéntico al training data) ──────────────────────
    "Consulta medica. Paciente masculino de 45 anios. Peso 82.0 kg, talla 170.5 cm.",
    "Px femenino, 67 anios. PA: 160 / 95, glucemia capilar 310 mg/dL.",
    "Presion arterial 125 / 82 mmHg. Glucosa 98 mg/dl. Temperatura 37.2C. FC 91 lpm.",
    "Se atiende a senora de 55 anios. Lleva 4 dias asi.",
    "Signos vitales: presion 118 sobre 74, glucosa 89, temperatura 38.5, pulso 102.",
    "Paciente masculino, edad: 2 anios. Tension arterial 100/65 mmHg.",
    "REPORTE DE ENFERMERIA. Paciente: masculino, 54 anios. Peso: 80.5 kg.",
    "PA 128/76. Glucosa 96. T 36.7. FC 77. Px varon de 33 anios.",
    "Edad: 58 anios, sexo masculino. Medidas: 74 kg / 166 cm.",
    "Nino de 5 anios, masculino. Fiebre 38.4C de 3 dias.",

    # ── Grupo 2: Con acentos españoles (llegan desde STT real) ───────────────
    "Paciente masculino de 45 años de edad. Presión arterial 125/82 mmHg.",
    "Señora de 55 años. Presión 170 sobre 100. Azúcar bien, 95.",
    "Temperatura de 37.2°C, frecuencia cardíaca de 91 latidos por minuto.",
    "Se atiende a niña de 11 años. Glucosa 88 mg/dL. Evolución 3 días.",
    "Médico registra: tensión 148/92 mmHg, glucemia 210 mg/dl.",
    "Consulta médica. Presión arterial de 125/82 mmHg, glucosa de 98 mg/dL.",
    "Paciente femenino de 26 años. Talla 1,58 m. Peso 55,5 kg.",
    "Señor de 71 años. Calentura 38.1 grados. Evolución de 7 días.",
    "Niño masculino de 7 años. Peso 22 kg, altura 118 cm. FC 115 bpm.",
    "Adulto mayor femenino, 78 años. Tensión 168/102. Calentura 37.8.",

    # ── Grupo 3: Números decimales con coma (formato europeo/regional) ────────
    "Temperatura 36,8 grados. Peso 63,5 kg. Talla 155,0 cm.",
    "Glucosa capilar 310,5 mg/dl. FC 88 lpm. PA 160/95.",
    "Paciente de 32,0 años. Glucemia 89,0. Temperatura 38,5 grados.",
    "Peso 82,0 kg. Talla 170,5 cm. Tensión 125/82 mmHg.",

    # ── Grupo 4: Abreviaciones y signos especiales médicos ────────────────────
    "PA: 160/95 mmHg. T: 36.8°C. FC: 88 lpm. Px femenino.",
    "TA 120/80. Glx 98. T 37.2°C. FC 91. Ev 5d. Masculino 45a.",
    "Signos vitales: TA 110/70, T 36.5°C, FC 72, Glx 85 mg/dl.",
    "Cuadro: Infeccioso/Vectorial. Dengue sospechoso. 4 días evolución.",
    "Categoría: Gastrointestinal. Duración: 5 días. PA 118/74.",

    # ── Grupo 5: Frases incompletas / campos faltantes ────────────────────────
    "Paciente masculino de 42 anios. Solo consulta por tos.",
    "Temperatura 38.5. Pulso 85. No se tomó presión.",
    "PA 160/95. Glucosa 310. Sin datos de peso y talla.",
    "Señora de 65 anios. Tensión alta. No coopera.",
    "Recién nacido. Peso 3.2 kg. Talla 49 cm. T 36.8°C. FC 140.",
    "Masculino 15 anios. Talla 165 cm, peso 55 kg. T 39.0. Pulso 105.",
]

assert len(FRASES) == 35, f"Se esperan 35 frases, hay {len(FRASES)}"

# Generar golden set
golden = []
for frase in FRASES:
    tok = tokenizar(frase)
    golden.append({
        "original": frase,
        "tokens":   tok,
        "ids":      ids(tok),
    })

out_path = Path("models/tokenizer_golden_set.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"frases": golden}, f, ensure_ascii=False, indent=2)

print(f"Golden set generado: {len(golden)} frases -> {out_path}")
print()

# Mostrar primeras 5 para inspección visual
for i, g in enumerate(golden[:5]):
    print(f"[{i+1}] {g['original'][:70]}")
    print(f"     Tokens: {g['tokens']}")
    print(f"     IDs:    {g['ids']}")
    print()

# Estadísticas
all_tokens = [t for g in golden for t in g["tokens"]]
unks = sum(1 for g in golden for i in g["ids"] if i == VOCAB["<UNK>"])
print(f"Total tokens generados: {len(all_tokens)}")
print(f"Tokens <UNK>: {unks} ({unks/len(all_tokens)*100:.1f}%)")
unique_unks = set(t for g in golden for t, i in zip(g["tokens"], g["ids"]) if i == VOCAB["<UNK>"])
if unique_unks:
    print(f"Tokens únicos <UNK>: {sorted(unique_unks)}")
    print("  ^ Estos son tokens que el modelo vería como desconocidos")
    print("  (esperado para fragmentos de palabras con acento: 'presi','n','os', etc.)")
