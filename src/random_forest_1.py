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

# --- 1. FUNCIÓN DE LIMPIEZA Y VALIDACIÓN INTEGRAL ---
def clean_spotify_data(df):
    """
    Realiza toda la limpieza especificada:
    - Elimina variables irrelevantes.
    - Elimina registros con valores nulos (exceptuando texto si se desea, pero aquí se aplica global).
    - Filtra valores fuera de rango lógico.
    - Elimina variables categóricas con un único valor (constantes), protegiendo texto y target.
    """
    df_clean = df.copy()
    
    # A. Eliminar variables que no aportan información o causan data leakage
    cols_a_eliminar = [
        "artist_id", "track_id", "album_id", "track_href", 
        "analysis_url", "external_urls.spotify", "track_uri", 
        "type", "is_local", "disc_number", 
        "album_release_date_precision", "album_release_year"
    ]
    df_clean = df_clean.drop(columns=[col for col in cols_a_eliminar if col in df_clean.columns], errors='ignore')
    
    # B. Eliminar registros con al menos un valor nulo
    filas_iniciales = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"Registros eliminados por nulos: {filas_iniciales - len(df_clean)}")
    
    # C. Eliminar registros con variables numéricas fuera de rango
    rangos_validos = {
        "danceability": (0.0, 1.0),
        "energy": (0.0, 1.0),
        "speechiness": (0.0, 1.0),
        "acousticness": (0.0, 1.0),
        "instrumentalness": (0.0, 1.0),
        "liveness": (0.0, 1.0),
        "valence": (0.0, 1.0),
        "key": (0, 11),
        "mode": (0, 1),
        "tempo": (0.0, 300.0),
        "duration_ms": (5000, 1800000),
    }
    
    mask_validos = pd.Series(True, index=df_clean.index)
    for col, (min_val, max_val) in rangos_validos.items():
        if col in df_clean.columns:
            mask_validos &= (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
            
    filas_fuera_rango = len(df_clean) - mask_validos.sum()
    df_clean = df_clean[mask_validos]
    print(f"Registros eliminados por estar fuera de rango: {filas_fuera_rango}")
    
    # D. Revisar variables categóricas: eliminar las que tengan un único valor (constantes)
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    cols_a_dropear_cat = []
    for col in categorical_cols:
        # Excluimos 'artist_name', 'track_name' y 'album_name' de esta regla
        if col not in ['artist_name', 'track_name', 'album_name']:
            if df_clean[col].nunique() <= 1:
                cols_a_dropear_cat.append(col)
                
    if cols_a_dropear_cat:
        print(f"Eliminando variables categóricas con valor único: {cols_a_dropear_cat}")
        df_clean = df_clean.drop(columns=cols_a_dropear_cat)
        
    # Asegurar que las columnas de texto sean de tipo string
    for text_col in ['track_name', 'album_name']:
        if text_col in df_clean.columns:
            df_clean[text_col] = df_clean[text_col].astype(str)
            
    return df_clean

# --- 2. CARGA Y PREPARACIÓN DE DATOS ---
filepath = "data/DatosPractica3.csv" 
df_raw = pd.read_csv(filepath)

# Aplicar limpieza completa
df_clean = clean_spotify_data(df_raw)

# Mostrar la tabla de conteo de canciones por artista en la terminal
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