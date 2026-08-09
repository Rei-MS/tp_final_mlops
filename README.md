# TP Final MLOps I - CEIA
## Predicción de satisfacción de pasajeros de aerolíneas

### Integrantes

| SIU | Nombre | Email |
|---|---|---|
| a2529 | María Teresa Mallaupoma León | mtmallaupoma2376@gmail.com |
| a2318 | Reinaldo Magallanes Saunders | rei.magallanes@gmail.com |

---

## 1. Descripción del proyecto

Este trabajo práctico implementa un pipeline MLOps de punta a punta para predecir la satisfacción de pasajeros de aerolíneas.

A partir de datos demográficos, características del viaje y valoraciones de servicios, se entrena un modelo de clasificación que predice:

- `satisfied`
- `neutral or dissatisfied`

El objetivo es construir un flujo reproducible que incluya entrenamiento, optimización, tracking de experimentos, almacenamiento de artefactos, registro y versionado del modelo, orquestación con Airflow, serving mediante FastAPI y pruebas automatizadas.

## 2. Dataset

Se utiliza el dataset **Airline Passenger Satisfaction** de Kaggle:

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

```text
datasets/
└── aerolineas/
    ├── README.md
    ├── train.csv
    └── test.csv
```

| Dataset | Registros |
|---|---:|
| `train.csv` | 103,904 |
| `test.csv` | 25,976 |

Variable objetivo:

```text
satisfaction
```

Codificación:

```text
satisfied                -> 1
neutral or dissatisfied  -> 0
```

Las variables categóricas se transforman mediante one-hot encoding con `pandas.get_dummies()`. El conjunto de test se alinea con las columnas generadas a partir del conjunto de entrenamiento.

## 3. Modelo

Se utiliza `RandomForestClassifier`.

La optimización de hiperparámetros se realiza con **Optuna**. Cada trial evalúa una configuración mediante validación cruzada y utiliza F1-score como métrica de optimización.

Hiperparámetros explorados:

- `n_estimators`
- `criterion`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Para pruebas rápidas:

```env
OPTUNA_TRIALS=1
```

Para una ejecución final puede aumentarse, por ejemplo:

```env
OPTUNA_TRIALS=10
```

## 4. Arquitectura MLOps

```text
                         ┌──────────────┐
                         │   Airflow    │
                         │     DAG      │
                         └──────┬───────┘
                                │
                                ▼
                       python -m src.main
                                │
                                ▼
                    Random Forest + Optuna
                                │
                                ▼
                         ┌────────────┐
                         │   MLflow   │
                         └──────┬─────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
           PostgreSQL                       MinIO
      metadata / experimentos          artefactos / modelos
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                         Model Registry
                                │
                           alias @champion
                                │
                                ▼
                             FastAPI
                                │
                           POST /predict
                                │
                                ▼
                           Predicción
```

### Descriptivo

**Airflow:** orquesta el entrenamiento y ejecuta `python -m src.main`.

**MLflow:** registra runs, parámetros, métricas, artefactos y versiones del modelo.

**PostgreSQL:** backend store de MLflow.

**MinIO:** almacenamiento S3-compatible para artefactos.

**FastAPI:** carga el modelo `champion` y expone endpoints de inferencia.

## 5. Tecnologías y librerias utilizadas

- Python 3.11
- pandas
- NumPy
- scikit-learn
- Optuna
- MLflow
- PostgreSQL
- MinIO
- Apache Airflow
- FastAPI
- Uvicorn
- Docker
- Docker Compose
- pytest

## 6. Estructura del repositorio

```text
tp_final_mlops/
├── airflow/
│   ├── Dockerfile
│   └── requirements.txt
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── dags/
│   └── train_model_dag.py
├── datasets/
│   └── aerolineas/
│       ├── README.md
│       ├── train.csv
│       └── test.csv
├── mlflow_system/
│   ├── docker-compose.yml
│   └── dockerfiles/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataload.py
│   ├── main.py
│   ├── metrics.py
│   ├── registry.py
│   ├── tracking.py
│   └── train.py
├── tests/
│   └── test_dataload.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 7. Variables de entorno

Crear `.env`:

```bash
cp .env.example .env
```

Ejemplo:

```env
PG_USER=postgres
PG_PASSWORD=postgres
PG_DATABASE=mlflow_db
PG_PORT=5432

MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
MINIO_PORT=9000
MINIO_PORT_UI=9001
MLFLOW_BUCKET_NAME=mlflow

MLFLOW_PORT=5000
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=airline-satisfaction-v2
MLFLOW_MODEL_NAME=airline-satisfaction-best-random-forest
MLFLOW_MODEL_ALIAS=champion

OPTUNA_TRIALS=1
```

> `.env` es local y no debe versionarse. Se versiona `.env.example`.

## 8. Requisitos

Para ejecutar el proyecto completo:

- Git
- Docker Desktop
- Docker Compose

Para ejecutar tests o entrenamiento desde el host se recomienda Python 3.11.

Ejemplo con Conda:

```bash
conda create -n tp-mlops python=3.11 -y
conda activate tp-mlops
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 9. Clonar y levantar la infraestructura

```bash
git clone https://github.com/Rei-MS/tp_final_mlops.git
cd tp_final_mlops
cp .env.example .env
```

Levantar todos los servicios:

```bash
docker compose \
  --env-file .env \
  -f mlflow_system/docker-compose.yml \
  up --build -d
```

Verificar:

```bash
docker compose \
  --env-file .env \
  -f mlflow_system/docker-compose.yml \
  ps
```

Interfaces:

| Servicio | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| FastAPI / Swagger | http://localhost:8000/docs |
| MinIO Console | http://localhost:9001 |

## 10. Entrenamiento manual

```bash
conda activate tp-mlops

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=airline-satisfaction-v2
export MLFLOW_MODEL_NAME=airline-satisfaction-best-random-forest
export OPTUNA_TRIALS=1

python -m src.main
```

Una ejecución correcta finaliza con:

```text
Training Successful
Registered Version: <version>
FastAPI URI: models:/airline-satisfaction-best-random-forest@champion
```

## 11. Entrenamiento con Airflow

DAG:

```text
airline_model_training
```

Tarea:

```text
train_random_forest
```

Abrir:

```text
http://localhost:8080
```

Usuario:

```text
admin
```

Obtener la contraseña:

```bash
docker exec airline-airflow \
  cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Desde la UI:

1. Buscar `airline_model_training`.
2. Habilitar el DAG si está pausado.
3. Presionar **Trigger**.
4. Esperar a que `train_random_forest` finalice en **Success**.

Prueba por CLI:

```bash
docker exec airline-airflow \
  airflow dags test \
  airline_model_training
```

Verificar que Airflow detecta el DAG:

```bash
docker exec airline-airflow airflow dags list
```

## 12. MLflow Tracking

Abrir:

```text
http://localhost:5000
```

Seleccionar **Model training** y luego:

```text
airline-satisfaction-v2
```

En **Training runs** se consultan:

- runs principales;
- trials de Optuna;
- parámetros;
- métricas;
- artefactos;
- gráficos de evaluación.

## 13. Model Registry

Modelo registrado:

```text
airline-satisfaction-best-random-forest
```

Cada entrenamiento genera una nueva versión.

El modelo seleccionado utiliza el alias:

```text
champion
```

FastAPI carga:

```text
models:/airline-satisfaction-best-random-forest@champion
```

### Si cambia `champion`

FastAPI carga el modelo durante el startup. Si Airflow registra una nueva versión y cambia `champion`, reiniciar la API:

```bash
docker compose \
  --env-file .env \
  -f mlflow_system/docker-compose.yml \
  restart api
```

Esperar unos segundos y verificar:

```bash
curl http://127.0.0.1:8000/health
```

## 14. FastAPI

Swagger:

```text
http://localhost:8000/docs
```

Endpoints:

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Información básica |
| GET | `/health` | Estado de la API y del modelo |
| POST | `/predict` | Predicción |

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "models:/airline-satisfaction-best-random-forest@champion"
}
```

## 15. Predicción desde Swagger

Esta es la forma más sencilla de usar el modelo sin modificar código.

1. Abrir `http://localhost:8000/docs`.
2. Desplegar `POST /predict`.
3. Presionar **Try it out**.
4. Copiar el JSON de ejemplo.
5. Presionar **Execute**.
6. Consultar el **Response body**.

### Ejemplo completo listo para copiar y pegar

```json
{
  "features": {
    "Unnamed: 0": 0,
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Age": 52,
    "Type of Travel": "Business travel",
    "Class": "Eco",
    "Flight Distance": 160,
    "Inflight wifi service": 5,
    "Departure/Arrival time convenient": 4,
    "Ease of Online booking": 3,
    "Gate location": 4,
    "Food and drink": 3,
    "Online boarding": 4,
    "Seat comfort": 3,
    "Inflight entertainment": 5,
    "On-board service": 5,
    "Leg room service": 5,
    "Baggage handling": 5,
    "Checkin service": 2,
    "Inflight service": 5,
    "Cleanliness": 5,
    "Departure Delay in Minutes": 50,
    "Arrival Delay in Minutes": 44.0
  }
}
```

No se envían `id` ni `satisfaction`: `id` no es una variable de entrada del modelo y `satisfaction` es la variable a predecir.

Ejemplo de respuesta:

```json
{
  "prediction": 1,
  "label": "satisfied",
  "probability_satisfied": 0.96,
  "model_uri": "models:/airline-satisfaction-best-random-forest@champion"
}
```

La probabilidad exacta puede variar si `champion` apunta a otra versión del modelo.

### Nota sobre `Unnamed: 0`

El dataset original contiene una columna de índice llamada `Unnamed: 0`. En la implementación actual forma parte del esquema esperado por el modelo/API, por eso aparece en el ejemplo.

Como mejora futura conviene eliminarla explícitamente del preprocesamiento y reentrenar.

## 16. Predicción con `curl`

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Unnamed: 0": 0,
      "Gender": "Female",
      "Customer Type": "Loyal Customer",
      "Age": 52,
      "Type of Travel": "Business travel",
      "Class": "Eco",
      "Flight Distance": 160,
      "Inflight wifi service": 5,
      "Departure/Arrival time convenient": 4,
      "Ease of Online booking": 3,
      "Gate location": 4,
      "Food and drink": 3,
      "Online boarding": 4,
      "Seat comfort": 3,
      "Inflight entertainment": 5,
      "On-board service": 5,
      "Leg room service": 5,
      "Baggage handling": 5,
      "Checkin service": 2,
      "Inflight service": 5,
      "Cleanliness": 5,
      "Departure Delay in Minutes": 50,
      "Arrival Delay in Minutes": 44.0
    }
  }'
```

## 17. Generar un ejemplo automáticamente desde `test.csv`

```bash
python - <<'PY'
import json
import pandas as pd

df = pd.read_csv("datasets/aerolineas/test.csv").dropna()

row = df.iloc[0].drop(
    labels=["id", "satisfaction"],
    errors="ignore",
)

payload = {
    "features": json.loads(row.to_json())
}

with open("/tmp/prediction.json", "w") as f:
    json.dump(payload, f, indent=2)

print(json.dumps(payload, indent=2))
PY
```

Enviar:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/prediction.json
```

## 18. Tests

```bash
python -m pytest -q
```

Resultado esperado:

```text
..
2 passed
```

Actualmente se verifica:

- codificación correcta del target;
- alineación de columnas entre train y test.

## 19. Flujo completo

```text
Docker Compose
     │
     ▼
Airflow
     │
     ▼
src.main
     │
     ▼
Optuna + Random Forest
     │
     ▼
MLflow Tracking
     │
     ▼
Model Registry
     │
     ▼
@champion
     │
     ▼
FastAPI
     │
     ▼
POST /predict
```

## 20. Estado del proyecto

| Componente | Estado |
|---|---|
| Dataset y preprocesamiento | ✅ |
| Alineación train/test | ✅ |
| Random Forest | ✅ |
| Optuna | ✅ |
| MLflow Tracking | ✅ |
| PostgreSQL | ✅ |
| MinIO | ✅ |
| Model Registry | ✅ |
| Alias `champion` | ✅ |
| Airflow DAG | ✅ |
| Entrenamiento desde Airflow | ✅ |
| Tests | ✅ |
| FastAPI | ✅ |
| `/health` | ✅ |
| `/predict` | ✅ |
| Swagger | ✅ |

## 21. Detener la infraestructura

```bash
docker compose \
  --env-file .env \
  -f mlflow_system/docker-compose.yml \
  down
```

> No utilizar `down -v` salvo que se quiera eliminar también la información persistida en los volúmenes.

## 22. Conclusiones

El proyecto implementa un flujo MLOps reproducible para clasificación de satisfacción de pasajeros.

Airflow permite ejecutar el entrenamiento de forma orquestada. MLflow centraliza experimentos, métricas, artefactos y versiones del modelo. PostgreSQL y MinIO soportan la persistencia de metadata y artefactos. El alias `champion` desacopla a FastAPI de un número de versión fijo y la API permite consumir el modelo mediante HTTP o desde Swagger.

El flujo implementado cubre:

```text
datos -> entrenamiento -> tracking -> registro -> orquestación -> serving
```

### Mejoras futuras

- Eliminar `Unnamed: 0` explícitamente.
- Encapsular preprocesamiento y modelo en un `sklearn.Pipeline`.
- Agregar tests específicos para FastAPI.
- Automatizar la recarga del modelo al cambiar `champion`.
- Incorporar CI/CD.
