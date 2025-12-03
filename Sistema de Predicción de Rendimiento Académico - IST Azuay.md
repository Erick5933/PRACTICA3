# Sistema de Predicción de Rendimiento Académico - IST Azuay

## Descripción General

Este proyecto implementa una **aplicación web interactiva** desarrollada con **Python y Streamlit** que predice el rendimiento académico de los estudiantes del Instituto Superior Tecnológico del Azuay (IST Azuay). La aplicación utiliza **modelos de aprendizaje automático** (Regresión Logística, Random Forest y XGBoost) para identificar tempranamente a los estudiantes en riesgo de no aprobar sus asignaturas.

## Características Principales

- **📊 Análisis Exploratorio de Datos (EDA):** Visualizaciones interactivas de tendencias académicas.
- **🤖 Comparación de Modelos:** Evaluación de tres algoritmos de clasificación con métricas detalladas.
- **🎯 Predictor en Tiempo Real:** Realiza predicciones para estudiantes individuales.
- **📈 Métricas de Rendimiento:** Incluye AUC-ROC, Precisión, Recall y F1-Score.
- **💡 Recomendaciones Personalizadas:** Sugerencias basadas en el resultado de la predicción.

## Estructura del Proyecto

```
academic_performance_app/
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
└── data/
    └── academic_performance_master.csv  # Dataset consolidado (no incluido)
```

## Requisitos Previos

- **Python 3.8 o superior**
- **pip** (gestor de paquetes de Python)

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd practica3
```

### 2. Crear un entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Preparación de Datos

Antes de ejecutar la aplicación, asegúrate de que el archivo de datos consolidado esté disponible:

1. Coloca el archivo `academic_performance_master.csv` en la raíz del proyecto o actualiza la ruta en `app.py` (línea 48).

El archivo debe contener las siguientes columnas:
- `Periodo`: Periodo académico (ej: "2020-2P")
- `Paralelo`: Paralelo de la clase
- `Identificacion_Estudiante`: Cédula del estudiante
- `Estudiante`: Nombre del estudiante
- `Carrera`: Nombre de la carrera
- `Nivel`: Nivel de estudio
- `Asignatura`: Nombre de la asignatura
- `Num_matricula`: Número de matrícula
- `Asistencia`: Porcentaje de asistencia (0-100)
- `Nota_final`: Nota final de la asignatura
- `Estado_Asignatura`: Estado (APROBADO, REPROBADO, RETIRADO, etc.)
- `Estado_Matricula`: Estado de la matrícula
- `Tipo_Ingreso`: Tipo de ingreso (NORMAL, etc.)
- `Cedula_docente`: Cédula del docente
- `Nombre_docente`: Nombre del docente

## Ejecución de la Aplicación

Para ejecutar la aplicación Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (por defecto en `http://localhost:8501`).

## Uso de la Aplicación

### 1. Inicio (📊)
- Visualiza estadísticas generales del dataset.
- Comprende el propósito y características del sistema.

### 2. Análisis Exploratorio (📈)
- **Distribución de Notas:** Histograma de la distribución de calificaciones.
- **Éxito por Periodo:** Tasa de aprobación en cada periodo académico.
- **Éxito por Carrera:** Comparación de tasas de éxito entre carreras.
- **Asistencia vs Nota:** Relación entre asistencia y rendimiento académico.

### 3. Comparación de Modelos (🤖)
- Tabla comparativa de métricas (AUC, Accuracy, Precisión, Recall, F1-Score).
- Gráfico de barras comparando AUC-ROC.
- Curvas ROC para visualizar el rendimiento de cada modelo.
- Identificación del mejor modelo (XGBoost).

### 4. Predictor en Tiempo Real (🎯)
- Ingresa datos del estudiante (asistencia, número de matrícula, carrera, periodo).
- Obtén una predicción de éxito/fracaso académico.
- Visualiza la probabilidad de aprobación.
- Recibe recomendaciones personalizadas.

## Modelos Implementados

### 1. Regresión Logística
- **AUC-ROC:** 0.8717
- **Ventajas:** Simple, interpretable, rápido.
- **Desventajas:** Rendimiento moderado.

### 2. Random Forest
- **AUC-ROC:** 0.8941
- **Ventajas:** Buen rendimiento, maneja no-linealidades.
- **Desventajas:** Menos interpretable que la regresión logística.

### 3. XGBoost (Mejor Modelo)
- **AUC-ROC:** 0.9157
- **Ventajas:** Excelente rendimiento, maneja desbalance de clases.
- **Desventajas:** Más complejo, requiere más tiempo de entrenamiento.

## Variables Predictoras

El modelo utiliza las siguientes características para realizar predicciones:

- **Asistencia:** Porcentaje de asistencia del estudiante (0-100%).
- **Número de Matrícula:** Número de veces que el estudiante ha cursado la asignatura.
- **Tipo de Ingreso:** Categoría de ingreso (NORMAL, etc.).
- **Carrera:** Programa académico en el que está inscrito.
- **Periodo:** Periodo académico en el que se realiza el curso.

## Interpretación de Resultados

### Predicción de Aprobación (✅)
Si el modelo predice que el estudiante aprobará:
- Mantener el nivel de asistencia actual.
- Continuar con las estrategias de estudio efectivas.

### Predicción de No Aprobación (❌)
Si el modelo predice que el estudiante no aprobará:
- **Aumentar la asistencia:** Es el predictor más importante.
- **Solicitar tutorías:** Buscar apoyo académico adicional.
- **Revisar métodos de estudio:** Considerar cambios en las estrategias.
- **Comunicarse con el docente:** Informar sobre dificultades.

## Métricas de Rendimiento

- **AUC-ROC:** Área bajo la curva ROC (0-1). Mide la capacidad de discriminación del modelo.
- **Accuracy:** Proporción de predicciones correctas.
- **Precisión:** Proporción de predicciones positivas correctas.
- **Recall:** Proporción de casos positivos identificados correctamente.
- **F1-Score:** Media armónica de precisión y recall.

## Limitaciones y Consideraciones

1. **Desbalance de Clases:** El dataset tiene más estudiantes aprobados que reprobados, lo que puede afectar el rendimiento en la clase minoritaria.
2. **Variables Disponibles:** El modelo utiliza solo las variables disponibles en el dataset. Incluir variables socioeconómicas o demográficas podría mejorar las predicciones.
3. **Datos Históricos:** Las predicciones se basan en patrones históricos. Cambios significativos en la institución podrían afectar la precisión.

## Trabajo Futuro

- Incluir variables socioeconómicas y demográficas.
- Implementar modelos de series temporales para predicciones a largo plazo.
- Desarrollar un dashboard de monitoreo en tiempo real.
- Integrar con el sistema de gestión académica del IST Azuay.
- Realizar validación cruzada más rigurosa.

## Contacto y Soporte

Para preguntas o sugerencias sobre este proyecto, por favor contacta al equipo de desarrollo.

## Licencia

Este proyecto es de uso interno del Instituto Superior Tecnológico del Azuay.

---

**Última actualización:** Octubre 2025

**Desarrollado con:**
- Python 3.11
- Streamlit 1.51.0
- Scikit-learn 1.7.2
- XGBoost 3.1.1
- Pandas 2.2.3
- Matplotlib 3.10.1
- Seaborn 0.13.2
