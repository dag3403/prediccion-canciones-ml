# 🎵 Clasificación de Artistas Musicales y Dashboard de Machine Learning

Pipeline completo de Machine Learning y dashboard interactivo diseñado para predecir el artista/cantante de una canción basándose en sus características de audio, metadatos de pistas y atributos de texto.

---

## 📋 Descripción General del Proyecto

El objetivo de este proyecto es construir, entrenar y evaluar una batería robusta de modelos de clasificación multiclase capaces de identificar el **artista** de un tema a partir exclusivamente de sus características de audio (obtenidas mediante la API de Spotify) y sus metadatos.

Para gestionar los desequilibrios de clases reales entre los diferentes artistas, el proyecto evalúa sistemáticamente **cuatro estrategias distintas de balanceo de datos y aprendizaje sensible a costos** a través de **nueve algoritmos punteros de Machine Learning**, guardando las métricas de rendimiento en archivos JSON estructurados y visualizándolas mediante un dashboard web interactivo (`app.py`).

---

## 🗂️ Estructura del Proyecto

```text
├── data/
│   └── DatosPractica3.csv        # Dataset extraído y limpiado de la API de Spotify
├── src/                          # Código fuente y pipelines de ML
│   ├── modelos_ganadores/        # Modelos con mejor rendimiento guardados (.json / checkpoints)
│   ├── catboost 2/
│   ├── hibrido 1/
│   ├── knn sobreajuste/
│   ├── lightgbm 1/
│   ├── naive bayes 1/
│   ├── randim forest 3/
│   ├── regresion logistica 4/
│   ├── xgboost 1/
│   └── SVM 1/
├── app.py                        # Dashboard interactivo en Streamlit para comparar modelos
├── requirements.txt              # Dependencias de Python
└── README.md                     # Documentación del proyecto