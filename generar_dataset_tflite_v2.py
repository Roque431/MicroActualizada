"""
FASE A + B — Dataset BIO v2 con diversidad real y set ciego de 20 frases.

Reutiliza hacer_texto y logica base de entrenar_ner.py.
Agrega 20 estilos nuevos con estructuras genuinamente distintas.
Embebe 20 frases ciegas escritas a mano (no derivadas de las 55 de prueba).
"""
import json, random, re
from collections import Counter
from pathlib import Path
import importlib.util, types

SEED = 42
random.seed(SEED)
rng = random.Random(SEED)

# ── Importar logica base de entrenar_ner.py ───────────────────────────────────
spec = importlib.util.spec_from_file_location("entrenar_ner", "entrenar_ner.py")
_src = spec.loader.get_source("entrenar_ner")
_mod = types.ModuleType("entrenar_ner")
exec(compile(_src[:_src.index("# ── Construir modelo")], "entrenar_ner.py", "exec"), _mod.__dict__)

hacer_texto = _mod.hacer_texto
fmt         = _mod.fmt
metros      = _mod.metros
ESTILOS_V1  = _mod.ESTILOS      # los 10 originales
DATASET_V1  = _mod.DATASET

# ── Vocabularios EXPANDIDOS ───────────────────────────────────────────────────
# Sinonimos que NO existian en las plantillas originales
SEXO_M_EXT  = ["masculino","hombre","varon","del sexo masculino","caballero",
               "adulto masculino","escolar masculino","paciente del sexo masculino"]
SEXO_F_EXT  = ["femenino","mujer","femenina","del sexo femenino","dama",
               "adulta femenina","escolar femenina","paciente del sexo femenino"]
SEXO_MC_EXT = ["senor","nino","muchacho","varon","caballero"]
SEXO_FC_EXT = ["senora","nina","muchacha","dama","senoras"]

PRESION_PREF = ["presion arterial ","tension arterial ","TA ","PA ","tension ","presion "]
PRESION_SEP  = [" / "," sobre "," x "]

GLUCOSA_PREF = ["glucosa ","azucar ","glucemia ","glicemia ",
                "nivel de azucar ","azucar en sangre ","glucemia capilar ","nivel de glucosa "]

TEMP_PREF    = ["temperatura ","calentura ","fiebre de ","T ","fiebre ",
                "temperatura corporal ","temperatura axilar "]

FC_PREF      = ["frecuencia cardiaca ","pulso ","FC ","ritmo cardiaco ","latidos ","ritmo "]

PESO_PREF    = ["peso ","pesa ","masa corporal ","peso corporal "]
PESO_SUFF    = [" kg"," kilogramos"," kilos"," kg de peso"]

TALLA_PREF   = ["talla ","mide ","estatura ","altura ","longitud "]
TALLA_SUFF   = [" cm"," centimetros"," m"]

DUR_PREF     = ["lleva ","desde hace ","presenta desde hace ","inicio hace ","cuadro de ","desde hace "]

INTRO_CONV   = ["Pues mire, ","Entonces, ","Le comento que ","Bueno, ",
                "Pues doctor, ","Mire usted, ","Como le decia, "]

# ── Funcion auxiliar: lista de partes dinamica ────────────────────────────────
def _pick(lst): return rng.choice(lst)

# Variantes del sufijo de edad — cubre "anios" (training artifact), "anos"
# (sin tilde, como se escribe en muchos teclados), y sin sufijo (solo el numero).
# Incluir las tres variantes obliga al modelo a aprender a detectar EDAD
# por posicion/contexto, no por memorizar un token especifico.
EDAD_SUFF = [" anios", " anos", " an", ""]   # vacio = sin sufijo
def _anios(): return rng.choice(EDAD_SUFF)

def _pa(r):
    """Retorna partes para presion arterial con separador aleatorio."""
    sep = _pick(PRESION_SEP)
    return [
        (_pick(PRESION_PREF), None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"),
        (sep, None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
    ]

def _temp(r):
    return [(_pick(TEMP_PREF), None), (fmt(r.temperatura_c), "TEMPERATURA")]

def _gluc(r):
    return [(_pick(GLUCOSA_PREF), None), (str(int(r.glucosa_mg_dl)), "GLUCOSA")]

def _fc(r):
    return [(_pick(FC_PREF), None), (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD")]

def _peso(r):
    pre = _pick(PESO_PREF); suf = _pick(PESO_SUFF)
    return [(pre, None), (fmt(r.peso_kg), "PESO_KG"), (suf, None)]

def _talla(r):
    pre = _pick(TALLA_PREF)
    if "m" in _pick(TALLA_SUFF) and rng.random() < 0.4:  # usar metros
        return [(pre, None), (metros(r.talla_cm), "TALLA_CM"), (" m", None)]
    return [(pre, None), (fmt(r.talla_cm), "TALLA_CM"), (" cm", None)]

def _dur(r):
    return [(_pick(DUR_PREF), None), (str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias", None)]

# ── 20 NUEVOS ESTILOS (e11-e30) ───────────────────────────────────────────────

def e11(r, rng_):
    """Sinonimos: tension/calentura/azucar, orden igual a SOAP."""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [("Px ", None), (sx, "SEXO"), (", ", None), (fmt(r.edad), "EDAD"), (_anios() + ". ", None)]
    partes += _pa(r) + [(". ", None)]
    partes += _gluc(r) + [(" mg/dl. ", None)]
    partes += _temp(r) + [(" grados. ", None)]
    partes += _fc(r) + [(" lpm. ", None)]
    partes += _peso(r) + [(", ", None)]
    partes += _talla(r) + [(". ", None)]
    partes += _dur(r) + [(".", None)]
    return hacer_texto(*partes)

def e12(r, rng_):
    """Conversacional con muletillas al inicio."""
    sx = _pick(SEXO_MC_EXT if r.sexo=="M" else SEXO_FC_EXT)
    intro = _pick(INTRO_CONV)
    art = "este " if r.sexo=="M" else "esta "
    partes = [
        (intro, None), (art, None), (sx, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ", pues lleva ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"),
        (" dias con lo mismo. ", None),
    ]
    partes += _pa(r) + [(". ", None)]
    partes += _temp(r) + [(". ", None)]
    partes += _gluc(r) + [(". El ", None)]
    partes += [(_pick(["pulso","ritmo","FC"]), None), (" en ", None),
               (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"), (".", None)]
    return hacer_texto(*partes)

def e13(r, rng_):
    """Formato de referencia medica: 'Le envio este paciente...'"""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Le envio este paciente ", None), (sx, "SEXO"),
        (" de ", None), (fmt(r.edad), "EDAD"), (_anios() + " de edad. ", None),
    ]
    partes += _talla(r) + [(", ", None)]
    partes += _peso(r) + [(". ", None)]
    partes += _pa(r) + [(" al ingreso. ", None)]
    partes += _gluc(r) + [(" mg/dl. ", None)]
    partes += _temp(r) + [(" grados. ", None)]
    partes += _fc(r) + [(" lpm.", None)]
    return hacer_texto(*partes)

def e14(r, rng_):
    """Ultra-abreviado sin puntuacion completa."""
    sx = "M" if r.sexo=="M" else "F"
    partes = [
        (fmt(r.edad), "EDAD"), ("/", None), (sx, "SEXO"), (" TA ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), ("/", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"), (" Glx ", None),
        (str(int(r.glucosa_mg_dl)), "GLUCOSA"), (" T ", None),
        (fmt(r.temperatura_c), "TEMPERATURA"), (" FC ", None),
        (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"), (" P ", None),
        (fmt(r.peso_kg), "PESO_KG"), (" Tall ", None),
        (fmt(r.talla_cm), "TALLA_CM"), (" Ev ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), ("d", None),
    ]
    return hacer_texto(*partes)

def e15(r, rng_):
    """Orden: signos vitales primero, demograficos al final."""
    partes = []
    partes += _pa(r) + [(". ", None)]
    partes += _gluc(r) + [(". ", None)]
    partes += _temp(r) + [(". ", None)]
    partes += _fc(r) + [(". Paciente ", None)]
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes += [(sx, "SEXO"), (" de ", None), (fmt(r.edad), "EDAD"), (_anios() + ". ", None)]
    partes += _peso(r) + [(", ", None)]
    partes += _talla(r) + [(". ", None)]
    partes += _dur(r) + [(".", None)]
    return hacer_texto(*partes)

def e16(r, rng_):
    """Tercera persona: 'El medico refiere...' """
    sx = _pick(SEXO_MC_EXT if r.sexo=="M" else SEXO_FC_EXT)
    partes = [
        ("El medico refiere que el ", None), (sx, "SEXO"),
        (" tiene ", None), (fmt(r.edad), "EDAD"), (_anios() + ". ", None),
        ("Registro de peso: ", None),
    ]
    partes += _peso(r) + [(". Talla: ", None)]
    partes += _talla(r) + [(". Tension: ", None)]
    partes += [(str(int(r.presion_sistolica)), "PRESION_SIS"), (" sobre ", None),
               (str(int(r.presion_diastolica)), "PRESION_DIA"), (". T: ", None)]
    partes += [(fmt(r.temperatura_c), "TEMPERATURA"), (" grados. Pulso: ", None)]
    partes += [(str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"), (" latidos.", None)]
    return hacer_texto(*partes)

def e17(r, rng_):
    """Datos en tabla/lista sin estructura narrativa."""
    sx = "masculino" if r.sexo=="M" else "femenino"
    partes = [
        ("Edad: ", None), (fmt(r.edad), "EDAD"), (_anios() + ". Sexo: ", None),
        (sx, "SEXO"), (". Peso: ", None), (fmt(r.peso_kg), "PESO_KG"),
        (" kg. Talla: ", None), (fmt(r.talla_cm), "TALLA_CM"),
        (" cm. Tension: ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), ("/", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (". Glucosa: ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (" mg/dl. Temperatura: ", None), (fmt(r.temperatura_c), "TEMPERATURA"),
        ("C. FC: ", None), (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (" lpm. Dias: ", None), (str(int(r.duracion_sintomas_dias)), "DURACION"),
        (".", None),
    ]
    return hacer_texto(*partes)

def e18(r, rng_):
    """Pediatrico con terminologia infantil especifica."""
    sx_ped = "masculino" if r.sexo=="M" else "femenino"
    tipo = rng.choice(["Nino", "Escolar", "Lactante", "Preescolar"])
    partes = [
        (tipo + " ", None), (sx_ped, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". Fiebre de ", None),
        (fmt(r.temperatura_c), "TEMPERATURA"), (" grados centígrados. FC ", None),
        (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"), (" lpm. ", None),
        ("Peso actual ", None), (fmt(r.peso_kg), "PESO_KG"), (" kg, mide ", None),
        (fmt(r.talla_cm), "TALLA_CM"), (" cm. Tension ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), ("/", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (". Glucemia ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (" mg/dl. Cuadro de ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias.", None),
    ]
    return hacer_texto(*partes)

def e19(r, rng_):
    """Sparse: solo signos vitales, sin peso/talla/categoria."""
    partes = [
        ("Signos vitales de paciente ", None),
        (_pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT), "SEXO"),
        (" de ", None), (fmt(r.edad), "EDAD"), (_anios() + ": ", None),
    ]
    partes += _pa(r) + [(", ", None)]
    partes += _temp(r) + [("C, ", None)]
    partes += _fc(r) + [(" lpm, ", None)]
    partes += _gluc(r) + [(" mg/dl.", None)]
    return hacer_texto(*partes)

def e20(r, rng_):
    """Sparse: solo datos demograficos y antropometricos."""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Paciente ", None), (sx, "SEXO"), (", edad ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". ", None),
    ]
    partes += _peso(r) + [(". ", None)]
    partes += _talla(r) + [(".", None)]
    return hacer_texto(*partes)

def e21(r, rng_):
    """Sparse: solo presion y glucosa (sin temperatura ni FC)."""
    sx = _pick(SEXO_MC_EXT if r.sexo=="M" else SEXO_FC_EXT)
    partes = [
        (_pick(INTRO_CONV), None), ("el/la ", None), (sx, "SEXO"),
        (" tiene ", None), (fmt(r.edad), "EDAD"), (_anios() + ". ", None),
    ]
    partes += _pa(r) + [(". ", None)]
    partes += _gluc(r) + [(" mg/dl. ", None)]
    partes += _dur(r) + [(".", None)]
    return hacer_texto(*partes)

def e22(r, rng_):
    """Urgencias: 'Paciente que llega con...'"""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Paciente ", None), (sx, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + " que llega con ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"),
        (" dias de cuadro. Al ingreso: ", None),
    ]
    partes += _pa(r) + [(", ", None)]
    partes += _temp(r) + [("C, ", None)]
    partes += _fc(r) + [(" lpm. Peso ", None)]
    partes += [(fmt(r.peso_kg), "PESO_KG"), (" kg.", None)]
    return hacer_texto(*partes)

def e23(r, rng_):
    """Con relleno irrelevante (mas tokens O)."""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Consulta general. Motivo: seguimiento. Paciente ", None),
        (sx, "SEXO"), (" de ", None), (fmt(r.edad), "EDAD"),
        (_anios() + " de edad, con antecedentes a revisar. ", None),
    ]
    partes += _peso(r) + [(" y talla ", None)]
    partes += [(fmt(r.talla_cm), "TALLA_CM"), (" cm. Sin alergias conocidas. ", None)]
    partes += _pa(r) + [(" mmHg. ", None)]
    partes += _gluc(r) + [(" mg/dl. ", None)]
    partes += _temp(r) + [(" grados. ", None)]
    partes += _fc(r) + [(" lpm. Evolucion de ", None)]
    partes += [(str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias. Se indica tratamiento.", None)]
    return hacer_texto(*partes)

def e24(r, rng_):
    """Con 'fiebre de X grados' como variante de temperatura."""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Se atiende ", None), (sx, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + " que refiere fiebre de ", None),
        (fmt(r.temperatura_c), "TEMPERATURA"), (" grados y ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias de malestar. ", None),
        ("La tension le salio ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), (" sobre ", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (". Pulso ", None), (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (". Azucar en sangre ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (". Peso ", None), (fmt(r.peso_kg), "PESO_KG"),
        (" kilos, estatura ", None), (fmt(r.talla_cm), "TALLA_CM"), (" cm.", None),
    ]
    return hacer_texto(*partes)

def e25(r, rng_):
    """Datos sin unidades explicitas, confiando solo en el contexto."""
    sx = _pick(SEXO_MC_EXT if r.sexo=="M" else SEXO_FC_EXT)
    partes = [
        (sx, "SEXO"), (" ", None), (fmt(r.edad), "EDAD"), (_anios() + ". Presion ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), (" sobre ", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"), (". Glucosa ", None),
        (str(int(r.glucosa_mg_dl)), "GLUCOSA"), (". Temperatura ", None),
        (fmt(r.temperatura_c), "TEMPERATURA"), (". Pulso ", None),
        (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"), (". Peso ", None),
        (fmt(r.peso_kg), "PESO_KG"), (". Talla ", None),
        (fmt(r.talla_cm), "TALLA_CM"), (". Dias ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), (".", None),
    ]
    return hacer_texto(*partes)

def e26(r, rng_):
    """Reporte de enfermeria: mas formal, campos con dos puntos."""
    sx = "masculino" if r.sexo == "M" else "femenino"
    cat = r.categoria_sintoma
    partes = [
        ("REPORTE DE ENFERMERIA. Paciente: ", None),
        (sx, "SEXO"), (", ", None), (fmt(r.edad), "EDAD"),
        (_anios() + ". Peso: ", None), (fmt(r.peso_kg), "PESO_KG"),
        (" kg. Talla: ", None), (fmt(r.talla_cm), "TALLA_CM"),
        (" cm. TA: ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), ("/", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (" mmHg. Temp: ", None), (fmt(r.temperatura_c), "TEMPERATURA"),
        (" C. FC: ", None), (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (" lpm. Glucosa: ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (" mg/dl. Tiempo evolucion: ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"),
        (" dias. Categoria: ", None), (cat, "CATEGORIA"), (".", None),
    ]
    return hacer_texto(*partes)

def e27(r, rng_):
    """Sparse aleatorio: descarta 3-5 campos para aprender ausencias."""
    campos_posibles = [
        ("sexo", lambda: [(_pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT), "SEXO")]),
        ("peso", lambda: _peso(r) + [(" corporal. ", None)]),
        ("talla", lambda: _talla(r) + [(". ", None)]),
        ("gluc", lambda: _gluc(r) + [(" mg/dl. ", None)]),
        ("temp", lambda: _temp(r) + [(" grados. ", None)]),
        ("fc", lambda: _fc(r) + [(" lpm. ", None)]),
        ("dur", lambda: _dur(r) + [(" de cuadro. ", None)]),
    ]
    sx_base = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Paciente ", None), (sx_base, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". ", None),
    ]
    partes += _pa(r) + [(". ", None)]
    # Agregar solo campos seleccionados aleatoriamente
    for nombre, gen_fn in campos_posibles:
        if rng.random() > 0.45:  # ~55% de probabilidad de incluir cada campo
            partes += gen_fn()
    return hacer_texto(*partes)

def e28(r, rng_):
    """Texto con glicemia/azucar como alternativa explicita."""
    sx = _pick(SEXO_MC_EXT if r.sexo=="M" else SEXO_FC_EXT)
    partes = [
        ("Atiende a ", None), (sx, "SEXO"), (" de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". Glicemia capilar en ayunas: ", None),
        (str(int(r.glucosa_mg_dl)), "GLUCOSA"), (" mg/dl. ", None),
        ("Tension arterial: ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), (" / ", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (" mmHg. Temperatura axilar: ", None),
        (fmt(r.temperatura_c), "TEMPERATURA"), (" C. ", None),
        ("Frecuencia del pulso: ", None),
        (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (" latidos por minuto. Tiempo de evolucion: ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias.", None),
    ]
    return hacer_texto(*partes)

def e29(r, rng_):
    """Presion como 'le salio X sobre Y' (estructura coloquial Chiapas)."""
    sx = _pick(SEXO_FC_EXT if r.sexo=="F" else SEXO_MC_EXT)
    art = "La " if r.sexo=="F" else "El "
    partes = [
        (art, None), (sx, "SEXO"), (" tiene ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". La tension le salio ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), (" sobre ", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"),
        (". La calentura en ", None), (fmt(r.temperatura_c), "TEMPERATURA"),
        (". El azucar en ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (". El pulso en ", None), (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (". Pesa ", None), (fmt(r.peso_kg), "PESO_KG"),
        (" kilos, mide ", None), (fmt(r.talla_cm), "TALLA_CM"),
        (" centimetros. Lleva ", None), (str(int(r.duracion_sintomas_dias)), "DURACION"),
        (" dias asi.", None),
    ]
    return hacer_texto(*partes)

def e30(r, rng_):
    """Presion en formato 'X x Y' (otra variante regional)."""
    sx = _pick(SEXO_M_EXT if r.sexo=="M" else SEXO_F_EXT)
    partes = [
        ("Paciente ", None), (sx, "SEXO"), (" con edad de ", None),
        (fmt(r.edad), "EDAD"), (_anios() + ". Tension ", None),
        (str(int(r.presion_sistolica)), "PRESION_SIS"), (" x ", None),
        (str(int(r.presion_diastolica)), "PRESION_DIA"), (" mmHg. ", None),
        ("Temperatura ", None), (fmt(r.temperatura_c), "TEMPERATURA"),
        (" grados. Azucar ", None), (str(int(r.glucosa_mg_dl)), "GLUCOSA"),
        (" mg/dl. Frecuencia cardiaca ", None),
        (str(int(r.frecuencia_cardiaca_bpm)), "FREC_CARD"),
        (" latidos. Peso ", None), (fmt(r.peso_kg), "PESO_KG"),
        (" kg. Talla ", None), (fmt(r.talla_cm), "TALLA_CM"),
        (" cm. Inicio hace ", None),
        (str(int(r.duracion_sintomas_dias)), "DURACION"), (" dias.", None),
    ]
    return hacer_texto(*partes)

ESTILOS_V2 = ESTILOS_V1 + [
    e11, e12, e13, e14, e15, e16, e17, e18, e19, e20,
    e21, e22, e23, e24, e25, e26, e27, e28, e29, e30,
]
print(f"Estilos totales: {len(ESTILOS_V2)}  (10 originales + 20 nuevos)")

# ── Importar el dataframe para regenerar ─────────────────────────────────────
import pandas as pd
df = pd.read_csv("consultas_clinicas.csv")
print(f"Filas CSV: {len(df):,}")

# ── Generar dataset ampliado ──────────────────────────────────────────────────
print("Generando dataset v2...")
DATASET_V2 = []
for _, row in df.iterrows():
    for fn in ESTILOS_V2:
        try:
            texto, ents = fn(row, rng)
            if texto and ents:
                DATASET_V2.append((texto, {"entities": ents}))
        except Exception:
            pass

print(f"  Dataset v2: {len(DATASET_V2):,} ejemplos")
print(f"  Promedio por fila: {len(DATASET_V2)/len(df):.1f} estilos")

# ── Tokenizador (identico a v1, debe mantenerse igual para Dart) ──────────────
_TOKEN_RE = re.compile(r'\d+[.,]\d+|\d+|[a-z]+')

def tokenizar(texto):
    texto_lower = texto.lower()
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(texto_lower)]

def spans_a_bio(tokens_con_pos, entidades):
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
            prev = char_label.get(t_ini - 1) if t_ini > 0 else None
            bio.append(f"B-{lbl}" if prev != lbl else f"I-{lbl}")
    return bio

# ── Construir ejemplos BIO ────────────────────────────────────────────────────
print("Construyendo ejemplos BIO...")
ejemplos_bio = []
for texto, ann in DATASET_V2:
    tok_pos = tokenizar(texto)
    if not tok_pos:
        continue
    tokens = [t for t, _, _ in tok_pos]
    labels = spans_a_bio(tok_pos, ann["entities"])
    assert len(tokens) == len(labels)
    ejemplos_bio.append({"tokens": tokens, "labels": labels})

# ── Vocabulario ───────────────────────────────────────────────────────────────
MIN_FREQ = 2
freq = Counter(tok for ej in ejemplos_bio for tok in ej["tokens"])
VOCAB = {"<PAD>": 0, "<UNK>": 1}
for tok, cnt in sorted(freq.items()):
    if cnt >= MIN_FREQ:
        VOCAB[tok] = len(VOCAB)

todas_labels = sorted({lbl for ej in ejemplos_bio for lbl in ej["labels"]})
LABEL2ID = {lbl: i for i, lbl in enumerate(todas_labels)}
O_ID = LABEL2ID["O"]

print(f"  Vocab v2: {len(VOCAB):,}  (v1 tenia 1,981)")
print(f"  Etiquetas: {len(LABEL2ID)}")

# Convertir a IDs
for ej in ejemplos_bio:
    ej["token_ids"] = [VOCAB.get(t, VOCAB["<UNK>"]) for t in ej["tokens"]]
    ej["label_ids"] = [LABEL2ID[l] for l in ej["labels"]]

# ── Split ─────────────────────────────────────────────────────────────────────
VAL_SPLIT = 0.10; TEST_SPLIT = 0.10
random.shuffle(ejemplos_bio)
n = len(ejemplos_bio)
n_test = int(n * TEST_SPLIT); n_val = int(n * VAL_SPLIT)
test_bio  = ejemplos_bio[:n_test]
val_bio   = ejemplos_bio[n_test:n_test+n_val]
train_bio = ejemplos_bio[n_test+n_val:]

# ── Guardar ───────────────────────────────────────────────────────────────────
OUT = Path("models")
with open(OUT/"tflite_vocab_v2.json",  "w", encoding="utf-8") as f: json.dump(VOCAB, f, ensure_ascii=False)
with open(OUT/"tflite_labels_v2.json", "w", encoding="utf-8") as f: json.dump(LABEL2ID, f, ensure_ascii=False)
for nombre, split in [("train_v2",train_bio),("val_v2",val_bio),("test_v2",test_bio)]:
    with open(OUT/f"tflite_{nombre}.json","w",encoding="utf-8") as f: json.dump(split, f, ensure_ascii=False)
    sz = (OUT/f"tflite_{nombre}.json").stat().st_size/1024
    print(f"  tflite_{nombre}.json — {len(split):,} ej ({sz:.0f} KB)")

print(f"\nDataset v2 listo: {n:,} ejemplos con {len(ESTILOS_V2)} estilos")
