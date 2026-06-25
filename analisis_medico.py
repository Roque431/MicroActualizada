import spacy
from spacy.matcher import Matcher

# 1. Cargamos el cerebro en español
nlp = spacy.load("es_core_news_sm")
matcher = Matcher(nlp.vocab)

# 2. Le enseñamos a la IA "patrones" de cómo hablan los médicos
# Buscaremos un sustantivo (como dolor o tos) seguido opcionalmente de un adjetivo (como intenso o seca)
patron_sintomas = [
    {"POS": "NOUN"}, 
    {"POS": "ADJ", "OP": "?"}  # El adjetivo es opcional (?)
]
matcher.add("SINTOMA_DETECTADO", [patron_sintomas])

# 3. La nota médica real dictada por la enfermera
nota_medica = "El paciente presenta fiebre alta, dolor de cabeza intenso y tos seca desde hace tres días."
doc = nlp(nota_medica)

print("\n=============================================")
print("🧠 EPIDIAGNOSTIX: EXTRACTOR DE DATOS MÉDICOS 🧠")
print("=============================================\n")

# 4. Ejecutamos el buscador inteligente
coincidencias = matcher(doc)

# 5. Imprimimos los síntomas reales estructurados
for match_id, start, end in coincidencias:
    span = doc[start:end]
    # Filtramos palabras de relleno para quedarnos solo con lo médico relevante
    if any(sintoma in span.text.lower() for sintoma in ["fiebre", "dolor", "tos"]):
        print(f"🚨 [SÍNTOMA IDENTIFICADO]: {span.text}")

print("\n---------------------------------------------")
print("✅ Datos listos para guardarse en SQLite local.")