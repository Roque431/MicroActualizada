import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

print("\n=============================================")
print(" EPIDIAGNOSTIX: ENTRENAMIENTO NLP (NER) ")
print("=============================================\n")

# 1. Creamos un modelo en blanco en español
nlp = spacy.blank("es")

# 2. DATASET DE ENTRENAMIENTO (Le enseñamos ejemplos y dónde están los datos exactos)
# Formato: ("Texto completo", {"entities": [(Inicio, Fin, "ETIQUETA")]})
DATOS_ENTRENAMIENTO = [
    ("El paciente tiene 45 años y refiere fiebre alta.", {"entities": [(18, 25, "EDAD"), (36, 47, "SINTOMA")]}),
    ("Femenina de 20 años presenta dolor de cabeza.", {"entities": [(12, 19, "EDAD"), (29, 44, "SINTOMA")]}),
    ("Niño de 5 años con tos seca desde ayer.", {"entities": [(8, 14, "EDAD"), (19, 27, "SINTOMA")]}),
    ("Hombre de 60 años con presion alta.", {"entities": [(10, 17, "EDAD"), (22, 34, "SINTOMA")]}),
    ("Paciente de 32 años manifiesta diarrea constante.", {"entities": [(12, 19, "EDAD"), (31, 48, "SINTOMA")]})
]

# 3. Añadimos el componente NER (Reconocimiento de Entidades) al cerebro en blanco
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)

# Añadimos las etiquetas que queremos que la IA aprenda a extraer solo con contexto
ner.add_label("EDAD")
ner.add_label("SINTOMA")

# 4. INICIAMOS EL ENTRENAMIENTO REAL
optimizer = nlp.begin_training()

print("🌀 Entrenando a la IA (Ajustando conexiones neuronales)...")
# Entrenaremos por 20 ciclos (Epochs) para que repita el conocimiento
for epoch in range(20):
    random.shuffle(DATOS_ENTRENAMIENTO)
    pérdida = {}
    
    for texto, anotaciones in DATOS_ENTRENAMIENTO:
        doc = nlp.make_doc(texto)
        ejemplo = Example.from_dict(doc, anotaciones)
        # Aquí la IA intenta adivinar, si se equivoca, el optimizer la corrige matemáticamente
        nlp.update([ejemplo], drop=0.2, sgd=optimizer, losses=pérdida)
    
    if epoch % 5 == 0:
        print(f"   Cycle (Epoch) {epoch} - Error del modelo: {pérdida['ner']:.4f}")

print("\n---------------------------------------------")
print(" ¡Entrenamiento completado localmente!")
print("---------------------------------------------\n")

# 5. PRUEBA DE FUEGO: Le damos una frase NUEVA que JAMÁS vio en el entrenamiento
frase_nueva = "Llega un adulto de 52 años con dolor de pecho fuerte."
doc_prueba = nlp(frase_nueva)

print(f"🔮 Texto procesado por la IA: '{frase_nueva}'")
print("Variables ordenadas extraídas por el modelo entrenado:")
for ent in doc_prueba.ents:
    print(f"   ↳ {ent.label_}: {ent.text}")