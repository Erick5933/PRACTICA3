import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import json
import os

# Cargar el DataFrame consolidado
data_path = "academic_performance_master.csv"
df = pd.read_csv(data_path)

# 1. Preparación de datos (Repetir pasos de codificación para asegurar consistencia)
df['Exito_Academico'] = df['Estado_Asignatura'].apply(lambda x: 1 if x == 'APROBADO' else 0)
df_model = df.copy()

# Codificación de variables categóricas (One-Hot Encoding)
df_model = pd.get_dummies(df_model, columns=['Tipo_Ingreso', 'Carrera', 'Periodo'], drop_first=True)

# Seleccionar características y variable objetivo
model_features = [col for col in df_model.columns if col.startswith(('Asistencia', 'Num_matricula', 'Tipo_Ingreso_', 'Carrera_', 'Periodo_'))]
df_final = df_model[model_features + ['Exito_Academico']].dropna()

X = df_final[model_features]
y = df_final['Exito_Academico']

# 2. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Definición y entrenamiento de modelos
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
}

results = {}
best_model_name = None
best_auc = -1

print("Iniciando entrenamiento y evaluación de modelos...")

for name, model in models.items():
    print(f"Entrenando {name}...")
    model.fit(X_train, y_train)
    
    # Predicción
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Evaluación
    auc_score = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "AUC": auc_score,
        "Report": report
    }
    
    if auc_score > best_auc:
        best_auc = auc_score
        best_model_name = name
        
    print(f"{name} - AUC: {auc_score:.4f}")

# 4. Guardar resultados de la comparación
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

# 5. Guardar el mejor modelo (solo el nombre)
with open("best_model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"\nComparación de modelos completada. El mejor modelo es: {best_model_name} con AUC: {best_auc:.4f}")
print("Resultados guardados en model_comparison_results.json")
print("Nombre del mejor modelo guardado en best_model_name.txt")
