import numpy as np
import pandas as pd

print("\n=============================================")
print("📈 EPIDIAGNOSTIX: MONITOR EPIDEMIOLÓGICO 📈")
print("=============================================\n")

# 1. Simulamos los datos históricos de una clínica rural en Chiapas (Casos de Dengue por semana)
# Las primeras 10 semanas todo es normal (entre 1 y 5 casos), pero en la semana 11 ocurre un pico (35 casos)
datos_clinica = {
    'Semana': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Casos_Dengue': [2, 1, 4, 3, 2, 5, 1, 3, 2, 4, 35, 3]  # <--- Nota el 35 en la semana 11
}

df = pd.DataFrame(datos_clinica)

# 2. El Algoritmo de Detección Matemática (Z-Score simplificado)
# Calculamos el promedio de casos normales y la desviación estándar
promedio = df['Casos_Dengue'].mean()
desviacion = df['Casos_Dengue'].std()

# Definimos un umbral: si un dato se aleja demasiado del promedio, es una anomalía
umbral = 1.5 

print(f"📊 Historial Analizado automáticamente.")
print(f"🔹 Promedio histórico de casos: {promedio:.2f}")
print("---------------------------------------------\n")

# 3. Analizamos semana por semana buscando picos atípicos
for index, row in df.iterrows():
    semana = int(row['Semana'])
    casos = int(row['Casos_Dengue'])
    
    # Si los casos rompen la tendencia normal
    if (casos - promedio) > (umbral * desviacion):
        print(f"🚨 ALERTAR JURISDICCIÓN: ¡Posible Brote en la Semana {semana}!")
        print(f"   ⚠️ Se detectaron {casos} casos de Dengue. Rompe el comportamiento normal de la zona.\n")
    else:
        print(f"🟢 Semana {semana:2d}: {casos:2d} casos - Comportamiento dentro del rango normal.")

print("\n---------------------------------------------")
print("✅ Análisis epidemiológico local terminado.")