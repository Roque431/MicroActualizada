"""Cuenta exitos y fallos por cada uno de los 30 estilos."""
import sys, io, pandas as pd, random, types, importlib.util
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

random.seed(42)
rng = random.Random(42)

# Cargar entrenar_ner.py
spec = importlib.util.spec_from_file_location("entrenar_ner", "entrenar_ner.py")
_src = spec.loader.get_source("entrenar_ner")
_cut = _src.index("Construir modelo")
_mod = types.ModuleType("entrenar_ner")
exec(compile(_src[:_cut], "entrenar_ner.py", "exec"), _mod.__dict__)
hacer_texto = _mod.hacer_texto; fmt = _mod.fmt; metros = _mod.metros
ESTILOS_V1 = _mod.ESTILOS

# Cargar estilos v2 directamente
import importlib.util as ilu
spec2 = ilu.spec_from_file_location("gen_v2", "generar_dataset_tflite_v2.py")
_src2 = spec2.loader.get_source("gen_v2")
_cut2 = _src2.index("ESTILOS_V2 = ")
_mod2 = types.ModuleType("gen_v2")
# Inyectar dependencias que el script necesita
_mod2.hacer_texto = hacer_texto
_mod2.fmt = fmt
_mod2.metros = metros
_mod2.rng = rng
_mod2.random = random
_mod2.pd = pd
exec(compile(_src2[:_cut2], "gen_v2.py", "exec"), _mod2.__dict__)

ESTILOS_NUEVOS = [_mod2.e11, _mod2.e12, _mod2.e13, _mod2.e14, _mod2.e15,
                  _mod2.e16, _mod2.e17, _mod2.e18, _mod2.e19, _mod2.e20,
                  _mod2.e21, _mod2.e22, _mod2.e23, _mod2.e24, _mod2.e25,
                  _mod2.e26, _mod2.e27, _mod2.e28, _mod2.e29, _mod2.e30]
TODOS = ESTILOS_V1 + ESTILOS_NUEVOS

df = pd.read_csv("consultas_clinicas.csv")
SAMPLE = df.head(200)

fallos = defaultdict(int)
exitos = defaultdict(int)
causas = defaultdict(set)

for _, row in SAMPLE.iterrows():
    for fn in TODOS:
        nombre = fn.__name__
        try:
            texto, ents = fn(row, rng)
            if texto and ents:
                exitos[nombre] += 1
            else:
                fallos[nombre] += 1
        except Exception as ex:
            fallos[nombre] += 1
            causas[nombre].add(type(ex).__name__ + ": " + str(ex)[:80])

print(f"{'Estilo':<8} {'OK':>5} {'FAIL':>6}  Causa")
print("-"*75)
for fn in TODOS:
    n = fn.__name__
    ok = exitos[n]; fail = fallos[n]
    c = list(causas.get(n, set()))
    c_str = c[0] if c else ""
    flag = "  <<< BUG" if fail > 0 else ""
    print(f"{n:<8} {ok:>5} {fail:>6}  {c_str}{flag}")

total_ok   = sum(exitos.values())
total_fail = sum(fallos.values())
print(f"\nTotal: {total_ok} OK  /  {total_fail} FAIL de {len(TODOS)*len(SAMPLE)} intentos")
print(f"Estilos con fallos: {sum(1 for n in fallos if fallos[n]>0)}/{len(TODOS)}")
