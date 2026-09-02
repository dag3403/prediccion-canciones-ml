import json
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # 1. Cargar datos
    filepath = "data/.csv"
    df_clean = pd.read_csv(filepath)


    # Asegurarnos de que las columnas de texto existan o combinarlas si es necesario
    # (Ejemplo: si tenemos 'track_name' y 'album_name', creamos un campo de texto combinado)
    if 'track_name' in df_clean.columns and 'album_name' in df_clean.columns:
        df_clean['text_combined'] = df_clean['track_name'].fillna('') + " " + df_clean['album_name'].fillna('')
    else:
        df_clean['text_combined'] = ""

    # 3. Separar Features (X) y Target (y)
    # Excluimos las columnas originales de texto sueltas y el ID de clase real
    cols_to_drop = ['artist_name', 'track_name', 'album_name', 'text_combined']
    X = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], errors='ignore')
    
    # Re-añadimos la columna de texto combinado específicamente para el procesamiento de lenguaje
    X['text_combined'] = df_clean['text_combined']
    y = df_clean['artist_name']

    # 4. Dividir datos: 90% entrenamiento, 10% test (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    # 5. Definición de estrategia de balanceo basada en el Cuantil 75%
    conteo_clases = y_train.value_counts()
    cuantil_75_clases = int(conteo_clases.quantile(0.75))
    
    sampling_strategy_dict = {
        clase: max(conteo, cuantil_75_clases) 
        for clase, conteo in conteo_clases.items() 
        if conteo < cuantil_75_clases
    }
    print(f"\nCuantil 75% seleccionado para balanceo híbrido: {cuantil_75_clases}")

    # 6. Configuración del ColumnTransformer Mixto (Numérico + Texto con SVD)
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    text_feature = 'text_combined'

    # Sub-pipeline para procesar el texto: TF-IDF seguido de TruncatedSVD para reducir dimensionalidad
    text_transformer = SklearnPipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=75, random_state=42)) # Reduce la matriz dispersa a 75 componentes densas
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )

    # Balanceo Híbrido (SMOTETomek)
    hibrido = SMOTETomek(sampling_strategy=sampling_strategy_dict, random_state=42)

    # Clasificador Random Forest
    base_rf = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1
    )

    # Pipeline completo usando ImbPipeline para manejar correctamente el remuestreo previo al modelo
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('resample', hibrido),
        ('classifier', base_rf)
    ])

    # 7. Optimización de hiperparámetros con 10-Fold CV
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    param_grid = {
        'classifier__n_estimators': [200, 300],
        'classifier__max_depth': [None, 20, 30],
        'classifier__min_samples_split': [2, 3]
    }

    print("\nIniciando optimización con GridSearchCV (10-Fold) y Pipeline Mixto...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")

    # 8. Evaluación en conjunto de test
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

    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- 9. GUARDAR RESULTADOS EN JSON ---
    os.makedirs("models", exist_ok=True)
    
    resultados_dict = {
        "best_parameters": grid_search.best_params_,
        "metrics": {
            "accuracy": float(acc),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_score_weighted": float(f1),
            "roc_auc_weighted": roc_auc_val,
            "oob_score": None,
            "oob_error": None
        },
        "confusion_matrix": conf_matrix.tolist()
    }

    json_path = "models/resultados_modelo_random_forest_4.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\nResultados guardados exitosamente en: {json_path}")

    # 10. Impresión de métricas en consola
    print("\n================ MÉTRICAS FINALES (PIPELINE MIXTO) ================")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC:       {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
    print("---------------------------------------------------------")
    print("Matriz de Confusión:\n", conf_matrix)
    print("---------------------------------------------------------")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred, zero_division=0))