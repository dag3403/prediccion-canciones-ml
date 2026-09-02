# 🎵 Clasificación de Artistas Musicales y Dashboard de Machine Learning

Pipeline completo de Machine Learning y dashboard interactivo diseñado para predecir el artista/cantante de una canción basándose en sus características de audio, metadatos de pistas y atributos de texto.

---

## 📋 Descripción General del Proyecto

El objetivo de este proyecto es construir, entrenar y evaluar una batería robusta de modelos de clasificación multiclase capaces de identificar el **artista** de un tema a partir exclusivamente de sus características de audio (obtenidas mediante la API de Spotify) y sus metadatos.

Para gestionar los desequilibrios de clases reales entre los diferentes artistas, el proyecto evalúa sistemáticamente **cuatro estrategias distintas de balanceo de datos y aprendizaje sensible a costos** a través de **nueve algoritmos punteros de Machine Learning**, guardando las métricas de rendimiento en archivos JSON estructurados dentro de la carpeta `models/` y visualizándolas mediante un dashboard web interactivo (`app.py`).

---

## 🗂️ Estructura del Proyecto

```text
├── predicción canciones/               # Directorio raíz del proyecto
│   ├── catboost_info/                  # Registros y archivos temporales generados automáticamente por CatBoost
│   │   ├── learn/                      # Métricas de aprendizaje por iteración
│   │   │   └── events.out.tfevents     # Eventos de registro para TensorBoard
│   │   └── tmp/                        # Archivos temporales de entrenamiento
│   │       ├── catboost_training.json  # Registro interno del proceso de entrenamiento
│   │       ├── learn_error.tsv         # Evolución del error de aprendizaje por iteración
│   │       └── time_left.tsv           # Estimación de tiempo restante de entrenamiento
│   ├── data/                           # Carpeta contenedora de datasets
│   │   └── DatosPractica3.csv          # Dataset principal extraído, depurado y listo para modelado
│   ├── models/                         # Carpeta central de resultados y métricas de los modelos (.json)
│   │   ├── resultados_catboost_1.json  # Métricas para CatBoost (Estrategia 1)
│   │   ├── resultados_catboost_2.json  # Métricas para CatBoost (Estrategia 2)
│   │   ├── resultados_catboost_3.json  # Métricas para CatBoost (Estrategia 3)
│   │   ├── resultados_catboost_4.json  # Métricas para CatBoost (Estrategia 4)
│   │   ├── resultados_knn_1.json       # Métricas para K-Nearest Neighbors (Estrategia 1)
│   │   ├── resultados_knn_2.json       # Métricas para K-Nearest Neighbors (Estrategia 2)
│   │   ├── resultados_knn_3.json       # Métricas para K-Nearest Neighbors (Estrategia 3)
│   │   ├── resultados_knn_4.json       # Métricas para K-Nearest Neighbors (Estrategia 4)
│   │   ├── resultados_lightgbm_1.json  # Métricas para LightGBM (Estrategia 1)
│   │   ├── resultados_lightgbm_2.json  # Métricas para LightGBM (Estrategia 2)
│   │   ├── resultados_lightgbm_3.json  # Métricas para LightGBM (Estrategia 3)
│   │   ├── resultados_lightgbm_4.json  # Métricas para LightGBM (Estrategia 4)
│   │   ├── resultados_lr_1.json        # Métricas para Regresión Logística (Estrategia 1)
│   │   ├── resultados_lr_2.json        # Métricas para Regresión Logística (Estrategia 2)
│   │   ├── resultados_lr_3.json        # Métricas para Regresión Logística (Estrategia 3)
│   │   ├── resultados_lr_4.json        # Métricas para Regresión Logística (Estrategia 4)
│   │   ├── resultados_mlp_1.json       # Métricas para Red Neuronal / MLP (Estrategia 1)
│   │   ├── resultados_mlp_2.json       # Métricas para Red Neuronal / MLP (Estrategia 2)
│   │   ├── resultados_mlp_3.json       # Métricas para Red Neuronal / MLP (Estrategia 3)
│   │   ├── resultados_mlp_4.json       # Métricas para Red Neuronal / MLP (Estrategia 4)
│   │   ├── resultados_modelo_random_forest_1.json # Métricas para Random Forest (Estrategia 1)
│   │   ├── resultados_modelo_random_forest_2.json # Métricas para Random Forest (Estrategia 2)
│   │   ├── resultados_modelo_random_forest_3.json # Métricas para Random Forest (Estrategia 3)
│   │   ├── resultados_modelo_random_forest_4.json # Métricas para Random Forest (Estrategia 4)
│   │   ├── resultados_naive_bayes_1.json        # Métricas para Naive Bayes (Estrategia 1)
│   │   ├── resultados_naive_bayes_2.json        # Métricas para Naive Bayes (Estrategia 2)
│   │   ├── resultados_naive_bayes_3.json        # Métricas para Naive Bayes (Estrategia 3)
│   │   ├── resultados_naive_bayes_4.json        # Métricas para Naive Bayes (Estrategia 4)
│   │   ├── resultados_svm_1.json       # Métricas para Support Vector Machines (Estrategia 1)
│   │   ├── resultados_svm_2.json       # Métricas para Support Vector Machines (Estrategia 2)
│   │   ├── resultados_svm_3.json       # Métricas para Support Vector Machines (Estrategia 3)
│   │   ├── resultados_svm_4.json       # Métricas para Support Vector Machines (Estrategia 4)
│   │   ├── resultados_xgboost_1.json   # Métricas para XGBoost (Estrategia 1)
│   │   ├── resultados_xgboost_2.json   # Métricas para XGBoost (Estrategia 2)
│   │   ├── resultados_xgboost_3.json   # Métricas para XGBoost (Estrategia 3)
│   │   └── resultados_xgboost_4.json   # Métricas para XGBoost (Estrategia 4)
│   ├── src/                            # Scripts de código fuente y análisis exploratorio
│   │   ├── .cache/                     # Caché interna del entorno de desarrollo
│   │   ├── análisis exploratorio.ipynb # Notebook de Jupyter para análisis estadístico y visualización inicial
│   │   ├── Catboost_1.py               # Pipeline de CatBoost (Estrategia 1)
│   │   ├── Catboost_2.py               # Pipeline de CatBoost (Estrategia 2)
│   │   ├── Catboost_3.py               # Pipeline de CatBoost (Estrategia 3)
│   │   ├── Catboost_4.py               # Pipeline de CatBoost (Estrategia 4)
│   │   ├── KNN_1.py                    # Pipeline de K-Nearest Neighbors (Estrategia 1)
│   │   ├── KNN_2.py                    # Pipeline de K-Nearest Neighbors (Estrategia 2)
│   │   ├── KNN_3.py                    # Pipeline de K-Nearest Neighbors (Estrategia 3)
│   │   ├── KNN_4.py                    # Pipeline de K-Nearest Neighbors (Estrategia 4)
│   │   ├── LightGBM_1.py               # Pipeline de LightGBM (Estrategia 1)
│   │   ├── LightGBM_2.py               # Pipeline de LightGBM (Estrategia 2)
│   │   ├── LightGBM_3.py               # Pipeline de LightGBM (Estrategia 3)
│   │   ├── LightGBM_4.py               # Pipeline de LightGBM (Estrategia 4)
│   │   ├── MLP_1.py                    # Pipeline de Red Neuronal / MLP (Estrategia 1)
│   │   ├── MLP_2.py                    # Pipeline de Red Neuronal / MLP (Estrategia 2)
│   │   ├── MLP_3.py                    # Pipeline de Red Neuronal / MLP (Estrategia 3)
│   │   ├── MLP_4.py                    # Pipeline de Red Neuronal / MLP (Estrategia 4)
│   │   ├── Naive_bayes_1.py            # Pipeline de Naive Bayes (Estrategia 1)
│   │   ├── Naive_bayes_2.py            # Pipeline de Naive Bayes (Estrategia 2)
│   │   ├── Naive_bayes_3.py            # Pipeline de Naive Bayes (Estrategia 3)
│   │   ├── Naive_bayes_4.py            # Pipeline de Naive Bayes (Estrategia 4)
│   │   ├── random_forest_1.py          # Pipeline de Random Forest (Estrategia 1)
│   │   ├── random_forest_2.py          # Pipeline de Random Forest (Estrategia 2)
│   │   ├── random_forest_3.py          # Pipeline de Random Forest (Estrategia 3)
│   │   ├── random_forest_4.py          # Pipeline de Random Forest (Estrategia 4)
│   │   ├── regresion_logistica_multiclase_1.py # Pipeline de Regresión Logística (Estrategia 1)
│   │   ├── regresion_logistica_multiclase_2.py # Pipeline de Regresión Logística (Estrategia 2)
│   │   ├── regresion_logistica_multiclase_3.py # Pipeline de Regresión Logística (Estrategia 3)
│   │   ├── regresion_logistica_multiclase_4.py # Pipeline de Regresión Logística (Estrategia 4)
│   │   ├── SVM_1.py                    # Pipeline de Support Vector Machines (Estrategia 1)
│   │   ├── SVM_2.py                    # Pipeline de Support Vector Machines (Estrategia 2)
│   │   ├── SVM_3.py                    # Pipeline de Support Vector Machines (Estrategia 3)
│   │   ├── SVM_4.py                    # Pipeline de Support Vector Machines (Estrategia 4)
│   │   ├── XGBOOST_1.py                # Pipeline de XGBoost (Estrategia 1)
│   │   ├── XGBOOST_2.py                # Pipeline de XGBoost (Estrategia 2)
│   │   ├── XGBOOST_3.py                # Pipeline de XGBoost (Estrategia 3)
│   │   └── XGBOOST_4.py                # Pipeline de XGBoost (Estrategia 4)
│   ├── .gitignore                      # Archivos y carpetas ignorados por control de versiones (Git)
│   ├── app.py                          # Aplicación y dashboard interactivo en Streamlit
│   ├── README.md                       # Documentación completa del proyecto
│   └── requirements.txt                # Listado de dependencias y librerías de Python necesarias
```

---

## 📊 Dataset e Ingeniería de Características

El dataset (`DatosPractica3.csv`) contiene canciones de diversos artistas, abarcando descriptores de audio y metadatos textuales:

* **Variable Objetivo (Target):** `artist_name` (El artista o cantante a predecir).
* **Características de Audio (Numéricas):** `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `loudness`.
* **Metadatos y Variables Categóricas:** `track_name`, `album_name`, `key_name`, `mode_name`, `key_mode`, `duration_ms`, `explicit`, `time_signature`, `track_number`.
* **Limpieza y Filtrado de Datos:**
* Eliminación de IDs aleatorios puros y enlaces técnicos (`artist_id`, `track_id`, `album_id`, `track_href`, `analysis_url`, `external_urls.spotify`, `track_uri`, `type`, `is_local`, `disc_number`, `album_release_year`).
* Supresión de características constantes con varianza cero (ej. `album_type`).
* Validación de rangos numéricos y verificación de cero valores nulos (`NaN`).



---

## 🤖 Algoritmos y Estrategias de Entrenamiento

### 1. Algoritmos Evaluados

Cada modelo se implementó y entrenó bajo idénticas condiciones de datos:

1. **CatBoost**
2. **K-Nearest Neighbors (KNN)**
3. **LightGBM**
4. **Multilayer Perceptron (MLP / Redes Neuronales)**
5. **Naive Bayes**
6. **Random Forest**
7. **Regresión Logística Multiclase**
8. **Support Vector Machines (SVM)**
9. **XGBoost**

### 2. Estrategias de Balanceo y Aprendizaje Sensible a Costos

Para solucionar el desequilibrio entre artistas con catálogos de diferente tamaño, se implementaron 4 estrategias distintas por cada algoritmo:

* **Estrategia 1:** Línea base desbalanceada (distribución de entrenamiento estándar).
* **Estrategia 2:** Aprendizaje sensible a costos (penalizando con mayor peso los errores de clasificación en clases minoritarias mediante `class_weight`).
* **Estrategia 3:** Remuestreo de muestras hasta alcanzar la **mediana** de la frecuencia de clases.
* **Estrategia 4:** Remuestreo de muestras hasta alcanzar el **percentil 75** de la frecuencia de clases.

---

## 📈 Evaluación y Almacenamiento de Métricas

* Todas las ejecuciones de entrenamiento, puntuaciones de validación cruzada (cross-validation), precisión, recall y métricas F1 se exportaron automáticamente a **archivos JSON** estructurados dentro de la carpeta `models/`.
* Los resultados se pueden explorar, filtrar y comparar interactivamente usando el dashboard de Streamlit.

---

## 💻 Dashboard Interactivo (`app.py`)

Aplicación web interactiva desarrollada para visualizar comparativas de rendimiento entre modelos, matrices de confusión y el impacto de los hiperparámetros según la estrategia de balanceo.

### Cómo ejecutar el Dashboard

1. Instalar las dependencias:
```bash
pip install -r requirements.txt

```


2. Lanzar la aplicación de Streamlit:
```bash
streamlit run app.py

```



---

## 🚀 Mejoras Futuras

* Integrar análisis de sentimiento de letras usando transformers de NLP (BERT / TF-IDF).
* Ampliar la representación de artistas en géneros diversos.
* Implementar sintonización de hiperparámetros mediante Optuna.