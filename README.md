# DL Forecasting: Household Electric Power Consumption

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945dd6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

Proyecto modular de forecasting de series de tiempo para predicción de demanda eléctrica hogareña utilizando redes recurrentes (**RNN simple**, **LSTM** y **GRU**) implementadas en PyTorch bajo una arquitectura *Factory Pattern*.

El flujo integral implementa:
**Git + pre-commit** $\rightarrow$ **DVC (Remote Local)** $\rightarrow$ **Preprocesamiento & Feature Engineering** $\rightarrow$ **Entrenamiento Recurrente** $\rightarrow$ **MLflow Tracking & Registry** $\rightarrow$ **Exportación del Mejor Modelo** $\rightarrow$ **Inferencia Productiva en FastAPI y Docker Compose**.

---

## 1. Dataset y Metodología de Preprocesamiento

Se utiliza el dataset [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) del repositorio UCI (~2 millones de mediciones minute-by-minute entre diciembre de 2006 y noviembre de 2010).

- **Variable Objetivo**: `Global_active_power` (potencia activa global por hogar en kilovatios, kW).
- **Agregación Temporal**: Resampleo de minutos a **frecuencia horaria media (`1h`)**.
- **Imputación de Faltantes**: Se aplica **interpolación lineal** en intervalos faltantes breves y relleno en bordes (`bfill().ffill()`).
  > **Decisión Técnica:** El consumo de electricidad residencial varía de manera continua debido a cargas continuas (refrigeradores, standby) y rutinas horarias estables. La interpolación lineal modela la transición suave entre horas adyacentes sin introducir escalones artificiales ni sesgo distribucional futuro.
- **División Temporal (Split)**: Partición estrictamente **secuencial / temporal**:
  - **70% Entrenamiento (Train)**
  - **15% Validación (Val)**
  - **15% Prueba (Test)**
  *(Nunca aleatorio para evitar data leakage temporal)*.
- **Normalización (Scaler)**: `StandardScaler` (o `MinMaxScaler`) se ajusta **únicamente sobre el split de train**. El scaler entrenado se persiste en `artifacts/scaler.joblib` y se reutiliza para transformar validación, test e inferencia en producción.
- **Ventanas Deslizantes (Sliding Windows)**: Parámetros configurables en `params.yaml`:
  - `input_window`: 24 horas históricas pasadas.
  - `horizon`: 1 paso adelante (configurable a horizontes mayores $H \ge 1$).

---

## 2. Estructura del Repositorio

```text
DL_Forecasting/
├── .gitignore                      # Exclusiones de Git (data, artifacts, mlflow.db, etc.)
├── .pre-commit-config.yaml         # Hooks de calidad de código (ruff, trailing-space, etc.)
├── Dockerfile                      # Imagen para entrenamiento, pipeline y tests
├── Dockerfile.inference            # Imagen ligera de inferencia en CPU para FastAPI
├── docker-compose.yml              # Orquestación de servicios (mlflow, train, test, inference)
├── dvc.yaml                        # Pipeline DVC reproducible (ingest -> preprocess -> train -> evaluate)
├── params.yaml                     # Configuración centralizada de hiperparámetros y rutas
├── requirements.txt                # Dependencias completas del proyecto
├── requirements-infer.txt          # Dependencias mínimas y ligeras para servicio de inferencia
├── ingest.py                       # Etapa 1 DVC: Ingesta y descarga desde UCI
├── preprocess.py                   # Etapa 2 DVC: Limpieza, resampleo, split y escalado
├── train.py                        # Etapa 3 DVC: Entrenamiento de modelos con tracking MLflow
├── evaluate.py                     # Etapa 4 DVC: Evaluación en Test vs Baseline ingenuo y metrics.json
├── callbacks/
│   ├── __init__.py
│   └── early_stopping.py           # EarlyStopping con respaldo del mejor estado
├── datasets/
│   ├── __init__.py
│   ├── factory.py                  # Factory de datasets y dataloaders
│   └── uci_power.py                # PyTorch Dataset sobre ventanas preprocesadas
├── engine/
│   ├── __init__.py
│   ├── evaluator.py                # Evaluación de modelo y baseline ingenuo
│   ├── metrics.py                  # RMSE, MAE y Safe MAPE
│   └── trainer.py                  # Bucles de train_one_epoch, evaluate y fit
├── inference/
│   ├── __init__.py
│   ├── api.py                      # Microservicio FastAPI (/predict y /health)
│   └── predict.py                  # Funciones de inferencia y pronóstico autoregresivo
├── models/
│   ├── __init__.py
│   ├── base.py                     # BaseNN abstract base class
│   ├── factory.py                  # Factory de modelos ('rnn', 'lstm', 'gru')
│   ├── gru.py                      # GRUForecaster
│   ├── lstm.py                     # LSTMForecaster
│   └── rnn.py                      # RNNForecaster
├── serving/
│   ├── __init__.py
│   └── select_best_model.py        # Selección automática del mejor modelo en MLflow
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py             # Pruebas unitarias de dataset
│   ├── test_inference.py           # Pruebas de la API FastAPI y funciones de predicción
│   ├── test_model.py               # Pruebas de arquitecturas recurrentes y gradientes
│   ├── test_preprocessing.py       # Pruebas de split temporal y ausencia de leakage
│   └── test_training.py            # Pruebas del loop de fit, early stopping y baseline
├── tracking/
│   ├── __init__.py
│   └── mlflow_utils.py             # Setup MLflow, logging compatible de artefactos y params
└── utils/
    ├── __init__.py
    ├── device.py                   # Detección de acelerador (CUDA / CPU)
    └── ploting.py                  # Gráficos de entrenamiento y series de tiempo
```

---

## 3. Instalación Local

### 3.1. Clonar el repositorio y crear entorno virtual
```bash
git clone <URL_DEL_REPOSITORIO>
cd DL_Forecasting

# Con venv:
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias:
pip install -r requirements.txt
```

### 3.2. Configurar Pre-commit
```bash
pre-commit install
pre-commit run --all-files
```

---

## 4. Pipeline DVC (Data Version Control)

El pipeline de DVC desacopla y rastrea dependencias y salidas de cada etapa:
1. `ingest`: Descarga de UCI a `data/raw/household_power_consumption.txt`.
2. `preprocess`: Resampleo horario, split temporal y ventanas $\rightarrow$ `data/processed/`, `artifacts/scaler.joblib`.
3. `train`: Entrenamiento recurrente $\rightarrow$ `artifacts/best_model.pth`.
4. `evaluate`: Evaluación en test set comparada contra baseline ingenuo $\rightarrow$ `metrics.json`.

### 4.1. Configuración de Remote Local
Para configurar un remote de almacenamiento local en DVC:

```bash
# 1. Inicializar DVC (si no está inicializado)
dvc init

# 2. Crear directorio para el remote local y configurarlo
mkdir -p /tmp/dvc_remote
dvc remote add -d local_storage /tmp/dvc_remote

# 3. Reproducir todo el pipeline
dvc repro

# 4. Subir artefactos versionados al remote local
dvc push
```

Para consultar métricas generadas:
```bash
dvc metrics show
```

---

## 5. MLflow Tracking y Selección del Mejor Modelo

Los experimentos y ejecuciones se almacenan en una base de datos local SQLite (`sqlite:///mlflow.db`).

### 5.1. Entrenar comparando modelos (RNN, LSTM, GRU)
```bash
# Entrenar el modelo configurado en params.yaml (por defecto LSTM)
python train.py

# O entrenar todos los modelos para comparar en MLflow
python train.py --model all
```

### 5.2. Visualizar la interfaz de MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Abre en tu navegador: [http://localhost:5000](http://localhost:5000)

### 5.3. Selección y exportación del mejor modelo
El script `serving/select_best_model.py` consulta las ejecuciones en MLflow mediante `mlflow.search_runs`, selecciona la corrida con menor `val_rmse` y exporta el modelo y scaler a `models_export/best_model/`:

```bash
python serving/select_best_model.py
```

---

## 6. Inferencia Local y API FastAPI

### 6.1. Inferencia por CLI
```bash
python inference/predict.py --steps 24
```

### 6.2. Levantar la API FastAPI
```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8080 --reload
```

- Documentación interactiva Swagger: [http://localhost:8080/docs](http://localhost:8080/docs)
- Health check: `GET http://localhost:8080/health`
- Predicción: `POST http://localhost:8080/predict`

**Ejemplo de petición con cURL:**
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "input_window": [
         1.2, 1.4, 1.1, 0.9, 0.8, 1.3, 2.1, 2.5, 2.8, 2.4, 1.9, 1.5,
         1.3, 1.1, 1.0, 1.2, 1.8, 2.9, 3.2, 2.7, 2.2, 1.8, 1.5, 1.3
       ]
     }'
```

**Respuesta esperada:**
```json
{
  "predictions": [1.4285],
  "model_type": "lstm",
  "horizon": 1,
  "unit": "kW"
}
```

---

## 7. Ejecución con Docker y Docker Compose

La arquitectura con `docker compose` orquesta 4 servicios aislados con persistencia de volúmenes:

| Servicio | Función | Puerto / Detalle |
|---|---|---|
| `mlflow` | Servidor de tracking con backend SQLite y artefactos persistentes | `http://localhost:5000` |
| `train` | Corre ingesta, preprocesamiento, entrenamiento y evaluación | Salidas en volúmenes `./data`, `./artifacts` |
| `test` | Ejecuta la suite de pruebas unitarias (`pytest tests/`) | Reporte en consola |
| `inference` | Microservicio FastAPI ligero (CPU) | `http://localhost:8080` |

### Comandos de Docker Compose:
```bash
# 1. Levantar servidor de MLflow en background
docker compose up -d mlflow

# 2. Ejecutar entrenamiento completo y selección del mejor modelo
docker compose up train

# 3. Ejecutar los tests unitarios dentro del contenedor
docker compose run --rm test

# 4. Levantar el microservicio de inferencia en producción
docker compose up -d inference

# 5. Detener todos los servicios
docker compose down
```

---

## 8. Checklist de Verificación de Punta a Punta

Ejecuta este checklist secuencial para comprobar el funcionamiento íntegro de la solución:

- [ ] **1. Calidad y Formato de Código:**
  ```bash
  pre-commit run --all-files
  ```
- [ ] **2. Pipeline Reproducible DVC:**
  ```bash
  dvc repro
  ```
- [ ] **3. Suite de Pruebas Unitarias:**
  ```bash
  pytest -v
  ```
- [ ] **4. Selección y Exportación del Mejor Modelo:**
  ```bash
  python serving/select_best_model.py
  ```
- [ ] **5. Inferencia por CLI:**
  ```bash
  python inference/predict.py --steps 24
  ```
- [ ] **6. Docker Compose Completo:**
  ```bash
  docker compose up --build
  ```
- [ ] **7. Verificación de Endpoints en Inferencia:**
  ```bash
  curl http://localhost:8080/health
  ```

