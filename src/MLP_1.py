import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

def clean_spotify_data(df):
    df_clean = df.copy()
    cols_a_eliminar = ["artist_id", "track_id", "album_id", "track_href", "analysis_url", "external_urls.spotify", "track_uri", "type", "is_local", "disc_number", "album_release_date_precision", "album_release_year"]
    df_clean = df_clean.drop(columns=[col for col in cols_a_eliminar if col in df_clean.columns], errors='ignore').dropna()
    rangos_validos = {"danceability": (0.0, 1.0), "energy": (0.0, 1.0), "speechiness": (0.0, 1.0), "acousticness": (0.0, 1.0), "instrumentalness": (0.0, 1.0), "liveness": (0.0, 1.0), "valence": (0.0, 1.0), "key": (0, 11), "mode": (0, 1), "tempo": (0.0, 300.0), "duration_ms": (5000, 1800000)}
    mask_validos = pd.Series(True, index=df_clean.index)
    for col, (min_val, max_val) in rangos_validos.items():
        if col in df_clean.columns: mask_validos &= (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
    df_clean = df_clean[mask_validos]
    for text_col in ['track_name', 'album_name']:
        if text_col in df_clean.columns: df_clean[text_col] = df_clean[text_col].astype(str)
    return df_clean

filepath = "data/DatosPractica3.csv"
df_raw = pd.read_csv(filepath)
df_clean = clean_spotify_data(df_raw)

X = df_clean.drop(columns=['artist_name'], errors='ignore')
y = LabelEncoder().fit_transform(df_clean['artist_name'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

track_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ('svd', TruncatedSVD(n_components=50, random_state=42))])
album_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))), ('svd', TruncatedSVD(n_components=25, random_state=42))])

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('track_text', track_pipeline, 'track_name'),
    ('album_text', album_pipeline, 'album_name')
], remainder='drop')

mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, learning_rate='adaptive', random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', mlp_clf)])

param_grid = {
    'classifier__alpha': [0.0001, 0.001],
    'classifier__learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

os.makedirs("models", exist_ok=True)
resultados_dict = {
    "estrategia": "MLP Caso 1: Sin balancear",
    "best_parameters": grid_search.best_params_,
    "metrics": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "f1_score_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        "roc_auc_weighted": float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'))
    },
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}
with open("models/resultados_mlp_1.json", "w", encoding="utf-8") as f:
    json.dump(resultados_dict, f, indent=4, ensure_ascii=False)
print("MLP Caso 1 (Sin balancear) completado y guardado.")