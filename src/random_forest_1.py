import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)


# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
filepath = "data/data_clean.csv" 
df_clean = pd.read_csv(filepath)

conteo_artistas_df = df_clean['artist_name'].value_counts().reset_index()
conteo_artistas_df.columns = ['Artista', 'Número de Canciones']

print("\n================ TABLA: CONTEO DE CANCIONES POR ARTISTA ================")
print(conteo_artistas_df.to_string(index=False))
print("-------------------------------------------------------------------------")
print(conteo_artistas_df['Número de Canciones'].describe())

# Separar Features (X) incluyendo texto y Target (y)
X = df_clean.drop(columns=['artist_name'], errors='ignore')
y = df_clean['artist_name']

# Separar datos: 90% entrenamiento, 10% test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# --- 3. CONSTRUCCIÓN DEL PIPELINE DE MODELADO CON TEXTO ---
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Pipelines específicos para texto (TF-IDF + SVD para reducir dimensionalidad de forma eficiente)
track_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('svd', TruncatedSVD(n_components=50, random_state=42))
])

album_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ('svd', TruncatedSVD(n_components=25, random_state=42))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('track_text', track_pipeline, 'track_name'),
        ('album_text', album_pipeline, 'album_name')
    ],
    remainder='drop'
)

base_rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', base_rf)
])

# --- 4. OPTIMIZACIÓN DE HIPERPARÁMETROS CON 10-FOLD CV ---
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

print("\nIniciando optimización de hiperparámetros con GridSearchCV (10-Fold)...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Mejor modelo optimizado
best_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")

# --- 5. EVALUACIÓN Y MÉTRICAS EN EL CONJUNTO DE TEST ---
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

try:
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    roc_auc_val = float(roc_auc)
except Exception as e:
    roc_auc_val = None

oob_score = best_model.named_steps['classifier'].oob_score_
oob_error = 1.0 - oob_score

conf_matrix = confusion_matrix(y_test, y_pred)

# --- 6. GUARDAR RESULTADOS EN JSON DENTRO DE LA CARPETA data ---
resultados_dict = {
    "best_parameters": grid_search.best_params_,
    "metrics": {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "roc_auc_weighted": roc_auc_val,
        "oob_score": float(oob_score),
        "oob_error": float(oob_error)
    },
    "confusion_matrix": conf_matrix.tolist()
}

json_path = "models/resultados_modelo_random_forest_1.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultados_dict, f, indent=4, ensure_ascii=False)

print(f"\nResultados guardados exitosamente en: {json_path}")

# --- 7. IMPRESIÓN DE MÉTRICAS EN CONSOLA ---
print("\n================ MÉTRICAS FINALES EN TEST (MIXTO) ================")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-Score:      {f1:.4f}")
print(f"ROC-AUC:       {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print(f"OOB Error:     {oob_error:.4f} (OOB Score: {oob_score:.4f})")
print("---------------------------------------------------------")
print("Matriz de Confusión:\n", conf_matrix)
print("---------------------------------------------------------")
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, zero_division=0))