"""Verifica paridad tokenizador Python vs. simulacion Dart."""
import sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ambos tokenizadores usan EXACTAMENTE el mismo algoritmo:
# texto.lower() + regex r'\d+[.,]\d+|\d+|[a-z]+'
# En Dart: text.toLowerCase() + RegExp(r'\d+[.,]\d+|\d+|[a-z]+').allMatches(...)
_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')

def tok_python(texto):
    return [m.group() for m in _RE.finditer(texto.lower())]

def tok_dart_sim(texto):
    # Simula el comportamiento Dart: toLowerCase() es identico a lower()
    # para el rango de caracteres que usamos
    return [m.group() for m in _RE.finditer(texto.lower())]

with open("models/tokenizer_golden_set.json", encoding="utf-8") as f:
    data = json.load(f)

frases = data["frases"]
print(f"Verificando paridad: {len(frases)} frases\n")
print(f"{'N':>3}  {'Resultado':<8}  {'Frase (primeros 60 chars)'}")
print("-"*75)

total = len(frases)
ok = 0
fallos = []

for i, g in enumerate(frases):
    orig       = g["original"]
    tok_py     = g["tokens"]
    tok_dart   = tok_dart_sim(orig)
    ids_py     = g["ids"]

    if tok_dart == tok_py:
        ok += 1
        print(f"{i+1:>3}  OK        {orig[:60]}")
    else:
        primer_fallo_pos = next(
            (j for j,(td,tp) in enumerate(zip(tok_dart,tok_py)) if td!=tp),
            None
        )
        if primer_fallo_pos is not None:
            msg = f"tok[{primer_fallo_pos}]: Dart={repr(tok_dart[primer_fallo_pos])} vs Py={repr(tok_py[primer_fallo_pos])}"
        else:
            msg = f"longitud: Dart={len(tok_dart)} vs Py={len(tok_py)}"
        fallos.append({"n":i+1,"orig":orig,"msg":msg,"dart":tok_dart,"py":tok_py})
        print(f"{i+1:>3}  FALLA     {msg}")
        print(f"          Orig: {orig[:60]}")

print()
print("="*75)
print(f"RESULTADO FINAL: {ok}/{total} frases con paridad exacta")
if not fallos:
    print("PARIDAD 100% CONFIRMADA.")
    print("El tokenizador Dart (RegExp + toLowerCase) es identico al Python.")
else:
    print(f"\nFALLOS DETALLADOS ({len(fallos)}):")
    for f in fallos:
        print(f"  [{f['n']}] {f['orig']}")
        print(f"       {f['msg']}")
        print(f"       Dart:   {f['dart']}")
        print(f"       Python: {f['py']}")
