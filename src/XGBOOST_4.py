import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


filepath = "data/data_clean.csv" 
df_clean = pd.read_csv(filepath)

X = df_clean.drop(columns=['artist_name'], errors='ignore')
y_raw = df_clean['artist_name']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

conteo_clases = pd.Series(y_train).value_counts()
cuantil_75_clases = int(conteo_clases.quantile(0.75))
sampling_strategy_dict = {clase: max(conteo, cuantil_75_clases) for clase, conteo in conteo_clases.items() if conteo < cuantil_75_clases}

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
track_pipeline = SklearnPipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ('svd', TruncatedSVD(n_components=50, random_state=42))])
album_pipeline = SklearnPipeline([('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))), ('svd', TruncatedSVD(n_components=25, random_state=42))])

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('track_text', track_pipeline, 'track_name'),
    ('album_text', album_pipeline, 'album_name')
], remainder='drop')

hibrido = SMOTETomek(sampling_strategy=sampling_strategy_dict, random_state=42)
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), eval_metric='mlogloss', random_state=42, n_jobs=-1)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', hibrido),
    ('classifier', xgb_clf)
])

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1]}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

resultados_dict = {
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

with open("models/resultados_xgboost_4.json", "w", encoding="utf-8") as f:
    json.dump(resultados_dict, f, indent=4, ensure_ascii=False)
print("Pipeline XGBoost SMOTETomek Cuantil 75% completado.")