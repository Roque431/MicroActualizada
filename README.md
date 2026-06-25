# EpiDiagnostix-Mayab — Componente ML

Microservicio de Machine Learning para detección de anomalías epidemiológicas
y extracción de variables clínicas desde texto libre en español.

**Plataforma offline-first** para personal de salud en comunidades rurales de Chiapas.

---

## Componentes

### 1. Extractor NER (spaCy — modelo propio entrenado)
Modelo NER entrenado localmente que convierte transcripciones de voz médico-paciente
en variables clínicas estructuradas. **100% offline, sin APIs de pago.**

- Entrenado con ~47,500 ejemplos generados de `consultas_clinicas.csv`
- 11 variables clínicas: edad, sexo, peso, talla, presión, glucosa, temperatura, FC, duración, categoría
- Generaliza por contexto — no depende de reglas rígidas
- Tamaño del modelo: ~10 MB

### 2. Isolation Forest (scikit-learn)
Detecta anomalías epidemiológicas a partir de las variables clínicas estructuradas.
- Entrenado con 4,752 consultas históricas
- F1 = 0.41 (mejor contamination = 0.02)
- Completamente offline

---

## Arquitectura Hexagonal

```
domain/          → Entidades puras + Puertos (interfaces)
application/     → Casos de uso (ExtraerVariables, DetectarAnomalia, ConsultarHistorial)
infrastructure/
  adapters/      → NERExtractorAdapter, IsolationForestAdapter, PostgresRepository
  api/           → FastAPI + Pydantic (routers, schemas)
  db/            → SQLAlchemy (tabla inferencias con JSONB)
microservicio/   → Dockerfile + docker-compose (API + PostgreSQL)
notebooks/       → Entrenamiento y explicación de modelos
```

---

## Endpoints REST

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Healthcheck |
| `POST` | `/extraccion` | Texto libre → variables clínicas |
| `POST` | `/deteccion-anomalias` | Variables → predicción Isolation Forest |
| `POST` | `/consulta-completa` | Extracción + detección en una llamada |
| `GET` | `/inferencias` | Historial persistido en PostgreSQL |

---

## Levantar el sistema

```bash
# 1. Copiar variables de entorno
cp microservicio/.env.example microservicio/.env
# Editar POSTGRES_PASSWORD en .env

# 2. Entrenar modelos (si no existen)
python entrenar_ner.py
# Los notebooks de entrenamiento están en notebooks/

# 3. Levantar con Docker
cd microservicio
docker-compose up --build
```

API disponible en `http://localhost:8000`  
Swagger: `http://localhost:8000/docs`

---

## Prueba de voz

```bash
# Google Speech (necesita internet solo para la transcripción)
python prueba_voz.py --api

# Modo offline (Whisper local)
python prueba_voz.py --whisper --api
```

---

## Tecnologías

- **Python 3.11** — FastAPI, spaCy 3.8, scikit-learn, SQLAlchemy
- **PostgreSQL 16** — Persistencia con JSONB
- **Docker Compose** — Despliegue en un solo comando
- **spaCy NER** — Modelo propio entrenado (sin APIs externas)

---

## Notebooks de estudio

| Notebook | Contenido |
|---|---|
| `00_explicacion_entrenamiento.ipynb` | Qué hace cada modelo y por qué |
| `01_isolation_forest.ipynb` | Entrenamiento del detector de anomalías |
| `02_extractor_llm_ner.ipynb` | Evaluación del extractor de variables |
| `03_explicacion_api.ipynb` | Cómo funciona la API y Docker |
| `04_entrenar_ner_local.ipynb` | Entrenamiento del modelo NER local |
