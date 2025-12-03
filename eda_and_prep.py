import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuración para gráficos
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Cargar el DataFrame consolidado
data_path = "academic_performance_master.csv"
df = pd.read_csv(data_path)

# 1. Limpieza y preparación de datos
# Convertir Periodo a categórico ordenado si es posible, o a string para gráficos
df['Periodo'] = df['Periodo'].astype(str)

# Crear la variable objetivo: Éxito/Fracaso (Aprobado/Reprobado)
# Asumiremos que 'APROBADO' es éxito y cualquier otro estado (REPROBADO, RETIRADO, etc.) es fracaso.
# Esto es una simplificación, pero es un buen punto de partida para un modelo predictivo binario.
df['Exito_Academico'] = df['Estado_Asignatura'].apply(lambda x: 1 if x == 'APROBADO' else 0)

# 2. Análisis Exploratorio de Datos (EDA)

# a) Distribución de la Nota Final
plt.figure(figsize=(10, 6))
sns.histplot(df['Nota_final'], bins=20, kde=True)
plt.title('Distribución de la Nota Final')
plt.xlabel('Nota Final')
plt.ylabel('Frecuencia')
plt.savefig('nota_final_distribution.png')
plt.close()

# b) Tasa de Éxito Académico por Periodo
period_success = df.groupby('Periodo')['Exito_Academico'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Periodo', y='Exito_Academico', data=period_success)
plt.title('Tasa de Éxito Académico por Periodo')
plt.xlabel('Periodo Académico')
plt.ylabel('Tasa de Éxito (Proporción de Aprobados)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('success_rate_by_period.png')
plt.close()

# c) Relación entre Asistencia y Nota Final
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Asistencia', y='Nota_final', data=df.sample(n=5000, random_state=42)) # Muestra para mejor visualización
plt.title('Relación entre Asistencia y Nota Final (Muestra)')
plt.xlabel('Asistencia (%)')
plt.ylabel('Nota Final')
plt.savefig('attendance_vs_grade.png')
plt.close()

# d) Tasa de Éxito por Carrera (Top 10)
career_success = df.groupby('Carrera')['Exito_Academico'].agg(['mean', 'count']).reset_index()
career_success = career_success[career_success['count'] > 100].sort_values(by='mean', ascending=False).head(10)

plt.figure(figsize=(14, 7))
sns.barplot(x='mean', y='Carrera', data=career_success)
plt.title('Top 10 Carreras por Tasa de Éxito Académico')
plt.xlabel('Tasa de Éxito (Proporción de Aprobados)')
plt.ylabel('Carrera')
plt.tight_layout()
plt.savefig('success_rate_by_career.png')
plt.close()

# 3. Preparación para el Modelado Predictivo (Regresión Logística)

# Seleccionar características (variables predictoras)
features = ['Asistencia', 'Num_matricula', 'Tipo_Ingreso', 'Carrera', 'Periodo']
df_model = df.copy()

# Codificación de variables categóricas (One-Hot Encoding)
df_model = pd.get_dummies(df_model, columns=['Tipo_Ingreso', 'Carrera', 'Periodo'], drop_first=True)

# Seleccionar solo las columnas numéricas y las dummies
# La variable 'Num_matricula' es numérica, pero es un conteo, la trataremos como numérica.
# 'Asistencia' ya es numérica.
model_features = [col for col in df_model.columns if col.startswith(('Asistencia', 'Num_matricula', 'Tipo_Ingreso_', 'Carrera_', 'Periodo_'))]

# Crear el DataFrame final para el modelado
df_final = df_model[model_features + ['Exito_Academico']].dropna()

# Guardar un resumen de las características para la propuesta
with open("model_features_summary.txt", "w") as f:
    f.write("Resumen de las características para el modelo predictivo:\n\n")
    f.write(f"Variable Objetivo: Exito_Academico (1=Aprobado, 0=Otro)\n")
    f.write(f"Características seleccionadas: {', '.join(model_features)}\n")
    f.write(f"Total de registros después de la preparación: {len(df_final)}\n")
    f.write("\nPrimeras 5 filas del DataFrame final:\n")
    f.write(df_final.head().to_markdown(index=False, numalign="left", stralign="left"))

print("Análisis Exploratorio de Datos completado. Gráficos guardados.")
print("Preparación de datos para el modelado completada. Resumen de características guardado.")
