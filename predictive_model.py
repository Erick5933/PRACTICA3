import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

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

# 3. Entrenamiento del modelo
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# 4. Evaluación del modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métricas
report = classification_report(y_test, y_pred, output_dict=True)
auc_score = roc_auc_score(y_test, y_proba)

# Coeficientes del modelo (para identificar predictores clave)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# 5. Generación de resultados para la propuesta

# Guardar el reporte de clasificación y el AUC
with open("model_performance_report.txt", "w") as f:
    f.write("## Reporte de Rendimiento del Modelo Predictivo (Regresión Logística)\n\n")
    f.write(f"**AUC-ROC Score:** {auc_score:.4f}\n\n")
    f.write("### Reporte de Clasificación:\n")
    f.write(pd.DataFrame(report).transpose().to_markdown(numalign="left", stralign="left"))
    f.write("\n\n### Coeficientes del Modelo (Top 10 Positivos y Negativos):\n")
    f.write("#### Predictores de Éxito (Coeficiente Positivo):\n")
    f.write(coefficients.head(10).to_markdown(index=False, numalign="left", stralign="left"))
    f.write("\n\n#### Predictores de Fracaso (Coeficiente Negativo):\n")
    f.write(coefficients.tail(10).to_markdown(index=False, numalign="left", stralign="left"))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC del Modelo de Rendimiento Académico')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

print("Modelado predictivo completado. Reporte de rendimiento y curva ROC guardados.")
