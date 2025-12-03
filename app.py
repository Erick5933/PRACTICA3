import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento Académico - IST Azuay",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">📚 Predicción de Rendimiento Académico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instituto Superior Tecnológico del Azuay</div>', unsafe_allow_html=True)

# Barra lateral de navegación
st.sidebar.title("🔍 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["📊 Inicio", "📈 Análisis Exploratorio", "🤖 Comparación de Modelos", "🎯 Predictor en Tiempo Real"]
)

# Cargar datos
@st.cache_data
def load_data():
    """Cargar el dataset consolidado."""
    df = pd.read_csv("academic_performance_master.csv")
    df['Exito_Academico'] = df['Estado_Asignatura'].apply(lambda x: 1 if x == 'APROBADO' else 0)
    return df

@st.cache_data
def prepare_data(df):
    """Preparar datos para el modelado."""
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=['Tipo_Ingreso', 'Carrera', 'Periodo'], drop_first=True)
    model_features = [col for col in df_model.columns if col.startswith(('Asistencia', 'Num_matricula', 'Tipo_Ingreso_', 'Carrera_', 'Periodo_'))]
    df_final = df_model[model_features + ['Exito_Academico']].dropna()
    return df_final, model_features

@st.cache_resource
def train_models(X_train, y_train):
    """Entrenar los modelos."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbose=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Cargar datos
df = load_data()
df_final, model_features = prepare_data(df)

# División de datos
X = df_final[model_features]
y = df_final['Exito_Academico']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar modelos
trained_models = train_models(X_train, y_train)

# Página: Inicio
if page == "📊 Inicio":
    st.header("Bienvenido al Sistema de Predicción de Rendimiento Académico")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", len(df), "Periodos Académicos: 7")
    
    with col2:
        tasa_exito = (df['Exito_Academico'].sum() / len(df)) * 100
        st.metric("Tasa de Éxito Promedio", f"{tasa_exito:.1f}%", "Estudiantes Aprobados")
    
    with col3:
        num_carreras = df['Carrera'].nunique()
        st.metric("Carreras Registradas", num_carreras, "Programas Académicos")
    
    st.markdown("---")
    
    st.subheader("📋 Descripción del Proyecto")
    st.write("""
    Este sistema utiliza **modelos de aprendizaje automático** para predecir el rendimiento académico 
    de los estudiantes del Instituto Superior Tecnológico del Azuay. El objetivo es identificar 
    tempranamente a los estudiantes en riesgo de no aprobar sus asignaturas, permitiendo implementar 
    estrategias de intervención y apoyo académico.
    
    **Características principales:**
    - 📊 Análisis exploratorio de datos con visualizaciones interactivas (Plotly)
    - 🤖 Comparación de tres modelos predictivos (Regresión Logística, Random Forest, XGBoost)
    - 🎯 Predictor en tiempo real para estudiantes individuales
    - 📈 Métricas de rendimiento detalladas
    """)
    
    st.markdown("---")
    
    st.subheader("🚀 Cómo Usar Esta Aplicación")
    st.write("""
    1. **📈 Análisis Exploratorio:** Explora los datos históricos y visualiza tendencias.
    2. **🤖 Comparación de Modelos:** Compara el rendimiento de diferentes algoritmos.
    3. **🎯 Predictor:** Realiza predicciones para estudiantes individuales.
    """)

# Página: Análisis Exploratorio
elif page == "📈 Análisis Exploratorio":
    st.header("📈 Análisis Exploratorio de Datos (EDA)")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["Distribución de Notas", "Éxito por Periodo", "Éxito por Carrera", "Asistencia vs Nota"])
    
    with tab1:
        st.subheader("Distribución de la Nota Final")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Nota_final'],
            nbinsx=20,
            name='Notas',
            marker=dict(color='#1f77b4', line=dict(color='#0d47a1', width=1))
        ))
        
        fig.update_layout(
            title='Distribución de la Nota Final',
            xaxis_title='Nota Final',
            yaxis_title='Frecuencia',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nota Promedio", f"{df['Nota_final'].mean():.2f}")
        with col2:
            st.metric("Nota Mínima", f"{df['Nota_final'].min():.2f}")
        with col3:
            st.metric("Nota Máxima", f"{df['Nota_final'].max():.2f}")
    
    with tab2:
        st.subheader("Tasa de Éxito Académico por Periodo")
        period_success = df.groupby('Periodo')['Exito_Academico'].agg(['mean', 'count']).reset_index()
        period_success.columns = ['Periodo', 'Tasa_Exito', 'Total_Registros']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=period_success['Periodo'],
            y=period_success['Tasa_Exito'],
            marker=dict(color=period_success['Tasa_Exito'], colorscale='Viridis', showscale=True),
            text=period_success['Tasa_Exito'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Tasa de Éxito: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Tasa de Éxito Académico por Periodo',
            xaxis_title='Periodo Académico',
            yaxis_title='Tasa de Éxito',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(period_success, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Tasa de Éxito por Carrera (Top 10)")
        career_success = df.groupby('Carrera')['Exito_Academico'].agg(['mean', 'count']).reset_index()
        career_success = career_success[career_success['count'] > 100].sort_values(by='mean', ascending=False).head(10)
        career_success.columns = ['Carrera', 'Tasa_Exito', 'Total_Registros']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=career_success['Carrera'],
            x=career_success['Tasa_Exito'],
            orientation='h',
            marker=dict(color=career_success['Tasa_Exito'], colorscale='Plasma', showscale=True),
            text=career_success['Tasa_Exito'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Tasa de Éxito: %{x:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Top 10 Carreras por Tasa de Éxito',
            xaxis_title='Tasa de Éxito',
            yaxis_title='Carrera',
            hovermode='y unified',
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(career_success, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Relación entre Asistencia y Nota Final")
        sample_df = df.sample(n=min(5000, len(df)), random_state=42)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_df['Asistencia'],
            y=sample_df['Nota_final'],
            mode='markers',
            marker=dict(
                size=5,
                color=sample_df['Nota_final'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Nota Final")
            ),
            text=sample_df['Estudiante'],
            hovertemplate='<b>Asistencia:</b> %{x}%<br><b>Nota Final:</b> %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Relación entre Asistencia y Nota Final (Muestra)',
            xaxis_title='Asistencia (%)',
            yaxis_title='Nota Final',
            hovermode='closest',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlación
        corr = df['Asistencia'].corr(df['Nota_final'])
        st.info(f"**Correlación de Pearson:** {corr:.4f} (Relación positiva fuerte)")

# Página: Comparación de Modelos
elif page == "🤖 Comparación de Modelos":
    st.header("🤖 Comparación de Modelos Predictivos")
    
    # Evaluar modelos
    results = {}
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "AUC": auc_score,
            "Accuracy": report['accuracy'],
            "Precision": report['1']['precision'],
            "Recall": report['1']['recall'],
            "F1": report['1']['f1-score'],
            "y_proba": y_proba
        }
    
    # Tabla de comparación
    st.subheader("📊 Tabla de Comparación de Modelos")
    comparison_df = pd.DataFrame({
        "Modelo": list(results.keys()),
        "AUC-ROC": [results[m]["AUC"] for m in results.keys()],
        "Accuracy": [results[m]["Accuracy"] for m in results.keys()],
        "Precisión": [results[m]["Precision"] for m in results.keys()],
        "Recall": [results[m]["Recall"] for m in results.keys()],
        "F1-Score": [results[m]["F1"] for m in results.keys()]
    })
    
    
    numeric_cols = comparison_df.select_dtypes(include=['float', 'int']).columns

    st.dataframe(
        comparison_df.style.format(
            {col: "{:.4f}" for col in numeric_cols}
        ),
        use_container_width=True,
        hide_index=True
    )

    # Gráfico de comparación de AUC (Plotly)
    st.subheader("📈 Comparación de AUC-ROC")
    models_list = list(results.keys())
    auc_scores = [results[m]["AUC"] for m in models_list]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_list,
        y=auc_scores,
        marker=dict(
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            line=dict(color='#000', width=2)
        ),
        text=[f'{score:.4f}' for score in auc_scores],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>AUC-ROC: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Comparación de AUC-ROC entre Modelos',
        yaxis_title='AUC-ROC Score',
        yaxis=dict(range=[0.8, 1.0]),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Curvas ROC (Plotly)
    st.subheader("📉 Curvas ROC")
    fig = go.Figure()
    
    for name, model in trained_models.items():
        y_proba = results[name]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results[name]["AUC"]
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{name} (AUC = {auc:.4f})',
            hovertemplate='<b>FPR:</b> %{x:.4f}<br><b>TPR:</b> %{y:.4f}<extra></extra>'
        ))
    
    # Línea de clasificador aleatorio
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Clasificador Aleatorio',
        line=dict(dash='dash', color='black'),
        hovertemplate='<b>FPR:</b> %{x:.4f}<br><b>TPR:</b> %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Curvas ROC - Comparación de Modelos',
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        hovermode='closest',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de métricas por clase
    st.subheader("📋 Métricas Detalladas por Clase (Clase 0 = No Aprobado, Clase 1 = Aprobado)")
    
    for name in results.keys():
        with st.expander(f"📌 {name}"):
            y_pred = trained_models[name].predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics_df = pd.DataFrame({
                'Métrica': ['Precisión', 'Recall', 'F1-Score', 'Soporte'],
                'Clase 0 (No Aprobado)': [
                    f"{report['0']['precision']:.4f}",
                    f"{report['0']['recall']:.4f}",
                    f"{report['0']['f1-score']:.4f}",
                    f"{int(report['0']['support'])}"
                ],
                'Clase 1 (Aprobado)': [
                    f"{report['1']['precision']:.4f}",
                    f"{report['1']['recall']:.4f}",
                    f"{report['1']['f1-score']:.4f}",
                    f"{int(report['1']['support'])}"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Selección del mejor modelo
    st.markdown("---")
    best_model_name = max(results, key=lambda x: results[x]["AUC"])
    st.success(f"✅ **Mejor Modelo Seleccionado:** {best_model_name} (AUC: {results[best_model_name]['AUC']:.4f})")

# Página: Predictor en Tiempo Real
elif page == "🎯 Predictor en Tiempo Real":
    st.header("🎯 Predictor de Rendimiento Académico")
    
    st.write("Ingresa los datos del estudiante para realizar una predicción de éxito académico.")
    
    # Usar el mejor modelo (XGBoost)
    best_model = trained_models["XGBoost"]
    
    # Inputs del usuario
    col1, col2 = st.columns(2)
    
    with col1:
        asistencia = st.slider("Asistencia (%)", 0, 100, 80)
        num_matricula = st.number_input("Número de Matrícula", 0, 10, 1)
    
    with col2:
        tipo_ingreso = st.selectbox("Tipo de Ingreso", df['Tipo_Ingreso'].unique())
        carrera = st.selectbox("Carrera", df['Carrera'].unique())
    
    periodo = st.selectbox("Periodo Académico", df['Periodo'].unique())
    
    # Preparar datos para predicción
    if st.button("🔮 Realizar Predicción", use_container_width=True):
        # Crear un DataFrame con los datos del estudiante
        input_data = pd.DataFrame({
            'Asistencia': [asistencia],
            'Num_matricula': [num_matricula],
            'Tipo_Ingreso': [tipo_ingreso],
            'Carrera': [carrera],
            'Periodo': [periodo]
        })
        
        # Codificar variables categóricas
        input_data = pd.get_dummies(input_data, columns=['Tipo_Ingreso', 'Carrera', 'Periodo'], drop_first=True)
        
        # Alinear con las características del modelo
        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[model_features]
        
        # Realizar predicción
        prediction_proba = best_model.predict_proba(input_data)[0]
        prediction = best_model.predict(input_data)[0]
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📊 Resultado de la Predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("✅ **Predicción: APROBADO**")
            else:
                st.error("❌ **Predicción: NO APROBADO**")
        
        with col2:
            st.metric("Probabilidad de Aprobación", f"{prediction_proba[1]*100:.1f}%")
        
        with col3:
            st.metric("Probabilidad de No Aprobación", f"{prediction_proba[0]*100:.1f}%")
        
        # Gráfico de probabilidades (Plotly)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['No Aprobado', 'Aprobado'],
            y=prediction_proba,
            marker=dict(color=['#d62728', '#2ca02c']),
            text=[f'{prob*100:.1f}%' for prob in prediction_proba],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probabilidad: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Distribución de Probabilidades',
            yaxis_title='Probabilidad',
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones")
        
        if prediction == 1:
            st.info("""
            ✅ El estudiante tiene una alta probabilidad de aprobar la asignatura.
            - Mantener el nivel de asistencia actual.
            - Continuar con las estrategias de estudio que han sido efectivas.
            """)
        else:
            st.warning("""
            ⚠️ El estudiante está en riesgo de no aprobar la asignatura.
            - **Aumentar la asistencia:** La asistencia es el predictor más importante.
            - **Solicitar tutorías:** Buscar apoyo académico adicional.
            - **Revisar métodos de estudio:** Considerar cambios en las estrategias de aprendizaje.
            - **Comunicarse con el docente:** Informar sobre dificultades y buscar orientación.
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>Sistema de Predicción de Rendimiento Académico - Instituto Superior Tecnológico del Azuay</p>
    <p>Desarrollado con Python, Scikit-learn, XGBoost, Streamlit y Plotly</p>
    </div>
""", unsafe_allow_html=True)
