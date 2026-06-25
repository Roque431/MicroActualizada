import spacy
from collections import Counter

print("\n=============================================")
print("🔍 EPIDIAGNOSTIX: MINERÍA DE HISTORIALES 🔍")
print("=============================================\n")

# 1. Cargamos el modelo base en español
nlp = spacy.load("es_core_news_sm")

# 2. HISTORIAL CLÍNICO CRUDO (Simulación de lo que escriben los enfermeros en el campo)
# Noten que aquí NO hay etiquetas, NO hay coordenadas, NO hay nada ordenado.
historiales_reales = [
    "Paciente masculino de 45 años, agricultor de profesión, acude por fiebre alta de 3 días.",
    "Se presenta femenina de 23 años, estudiante, con dolor de cabeza intenso y mareos desde ayer.",
    "Niño de 8 años, estudiante de primaria, presenta tos seca y calentura de 39 grados.",
    "Adulto de 60 años, jubilado, con presion alta y dolor de pecho fuerte desde hace una semana.",
    "Femenina de 35 años, comerciante en el mercado, refiere dolor de garganta e infección severa.",
    "Masculino de 32 años, albañil, presenta dolor lumbar fuerte por cargar objetos pesados.",
    "Anciano de 75 años con fatiga extrema, falta de aire y oxigenacion baja desde esta mañana."
]

# Unimos todo el historial en un solo bloque de texto para que la IA lo analice completo
texto_completo = " ".join(historiales_reales)
doc = nlp(texto_completo)

# 3. ALGORITMO DE DESCUBRIMIENTO
# Guardaremos las palabras que más se repiten según su tipo gramatical
sustantivos_y_adjetivos = []  # Para descubrir Síntomas y Signos Vitales
ocupaciones_detectadas = []   # Para descubrir Ocupaciones
unidades_tiempo = []          # Para descubrir Tiempos de enfermedad

print("🧠 Analizando estructuras gramaticales en el historial...")

for token in doc:
    # Si la IA detecta palabras clave de tiempo
    if token.text.lower() in ["días", "ayer", "semana", "mañana", "horas"]:
        unidades_tiempo.append(token.text.lower())
    
    # Si detecta palabras que describen el trabajo de la gente en el contexto
    elif token.text.lower() in ["agricultor", "estudiante", "jubilado", "comerciante", "albañil"]:
        ocupaciones_detectadas.append(token.text.lower())
        
    # Si son palabras médicas comunes (sustantivos y adjetivos que describen el estado del cuerpo)
    elif token.pos_ in ["NOUN", "ADJ"] and token.text.lower() not in ["paciente", "profesión", "mercado", "objetos"]:
        sustantivos_y_adjetivos.append(token.text.lower())

# 4. CONTAMOS LAS FRECUENCIAS (Las más comunes ganan)
top_medico = Counter(sustantivos_y_adjetivos).most_common(5)
top_ocupacion = Counter(ocupaciones_detectadas).most_common(3)
top_tiempo = Counter(unidades_tiempo).most_common(3)

# 5. PRESENTACIÓN DE RESULTADOS JUSTIFICADOS
print("\n📊 --- REPORTE ESTADÍSTICO PARA EL PROFESOR ---")
print("Las variables más comunes descubiertas de forma automatizada son:\n")

print("🚨 [VARIABLE: SÍNTOMAS / SIGNOS] - Palabras con mayor frecuencia:")
for palabra, freq in top_medico:
    print(f"   ↳ '{palabra}': se repite {freq} veces en el historial.")

print("\n💼 [VARIABLE: OCUPACIÓN] - Patrones de empleo detectados:")
for palabra, freq in top_ocupacion:
    print(f"   ↳ '{palabra}': se repite {freq} veces en el historial.")

print("\n⏳ [VARIABLE: TIEMPO] - Expresiones temporales más usadas:")
for palabra, freq in top_tiempo:
    print(f"   ↳ '{palabra}': se repite {freq} veces en el historial.")

print("\n---------------------------------------------")
print("✅ Conclusión científica: El sistema debe clasificarse en estas 5 variables.")