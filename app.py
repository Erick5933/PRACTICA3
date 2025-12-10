import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Rendimiento Académico - IST Azuay",
    page_icon="📚",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown('<div class="main-header">📚 Sistema de Análisis de Rendimiento Académico</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Instituto Superior Tecnológico del Azuay</p>', unsafe_allow_html=True)

# Navegación
st.sidebar.title("🔍 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Exploración de Datos", "🎯 Modelo Supervisado", "🔍 Modelo No Supervisado", "📈 Comparación"]
)

# ============================================================
# FUNCIONES DE CARGA Y PREPARACIÓN
# ============================================================

@st.cache_data
def load_data():
    """Cargar el dataset."""
    df = pd.read_csv("academic_performance_master.csv")
    # Crear variable objetivo: 1=APROBADO, 0=REPROBADO
    df['Aprobado'] = (df['Estado_Asignatura'] == 'APROBADO').astype(int)
    return df

@st.cache_data
def prepare_supervised_data(df):
    """Preparar datos para modelo supervisado."""
    # Seleccionar solo columnas relevantes
    features = ['Asistencia', 'Num_matricula']
    df_clean = df[features + ['Aprobado']].dropna()
    
    X = df_clean[features]
    y = df_clean['Aprobado']
    
    return X, y, features

@st.cache_data
def prepare_clustering_data(df):
    """Preparar datos para clustering."""
    df_cluster = df[['Asistencia', 'Nota_final']].dropna()
    return df_cluster

# Cargar datos
df = load_data()

# ============================================================
# PÁGINA: INICIO
# ============================================================

if page == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Análisis Académico")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total de Registros", f"{len(df):,}")
    
    with col2:
        tasa_aprobacion = (df['Aprobado'].sum() / len(df)) * 100
        st.metric("✅ Tasa de Aprobación", f"{tasa_aprobacion:.1f}%")
    
    with col3:
        st.metric("🎓 Carreras", df['Carrera'].nunique())
    
    st.markdown("---")
    
    st.subheader("📋 Objetivo del Proyecto")
    st.write("""
    Este sistema implementa **dos modelos de Machine Learning** para analizar el rendimiento académico:
    
    1. **Modelo Supervisado (Clasificación)**: Predice si un estudiante aprobará o reprobará
    2. **Modelo No Supervisado (Clustering)**: Agrupa estudiantes con patrones similares
    
    **Dataset**: `academic_performance_master.csv`  
    **Registros**: {:,} estudiantes  
    **Variables clave**: Asistencia, Nota Final, Carrera, Periodo
    """.format(len(df)))
    
    st.markdown("---")
    
    st.subheader("🎯 ¿Cómo usar esta aplicación?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Exploración de Datos**
        - Visualiza distribuciones
        - Identifica patrones
        - Estadísticas descriptivas
        """)
        
        st.success("""
        **🎯 Modelo Supervisado**
        - Regresión Logística
        - Predicción de aprobación
        - Matriz de confusión
        """)
    
    with col2:
        st.warning("""
        **🔍 Modelo No Supervisado**
        - K-Means Clustering
        - Agrupación de estudiantes
        - Perfiles académicos
        """)
        
        st.error("""
        **📈 Comparación**
        - Análisis de ambos modelos
        - Conclusiones
        - Recomendaciones
        """)

# ============================================================
# PÁGINA: EXPLORACIÓN DE DATOS
# ============================================================

elif page == "📊 Exploración de Datos":
    st.header("📊 Exploración de Datos")
    
    tab1, tab2, tab3 = st.tabs(["Vista General", "Distribuciones", "Correlaciones"])
    
    with tab1:
        st.subheader("Vista Previa del Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Información del Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dimensiones:**")
            st.write(f"- Filas: {len(df):,}")
            st.write(f"- Columnas: {len(df.columns)}")
            st.write(f"- Valores nulos: {df.isnull().sum().sum()}")
        
        with col2:
            st.write("**Tipos de Datos:**")
            st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Tipo', 'index': 'Columna'}))
        
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Distribución de Variables Clave")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de Nota Final
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Nota_final'].dropna(), bins=20, color='skyblue', edgecolor='black')
            ax.axvline(7, color='red', linestyle='--', linewidth=2, label='Nota Mínima (7.0)')
            ax.set_title('Distribución de Nota Final', fontsize=14, fontweight='bold')
            ax.set_xlabel('Nota Final')
            ax.set_ylabel('Frecuencia')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Métricas de Nota Final
            st.metric("Media", f"{df['Nota_final'].mean():.2f}")
            st.metric("Desviación Estándar", f"{df['Nota_final'].std():.2f}")
        
        with col2:
            # Distribución de Asistencia
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Asistencia'].dropna(), bins=20, color='lightgreen', edgecolor='black')
            ax.set_title('Distribución de Asistencia', fontsize=14, fontweight='bold')
            ax.set_xlabel('Asistencia (%)')
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Métricas de Asistencia
            st.metric("Media", f"{df['Asistencia'].mean():.1f}%")
            st.metric("Desviación Estándar", f"{df['Asistencia'].std():.1f}%")
        
        # Distribución Aprobados/Reprobados
        st.subheader("Distribución de Aprobados vs Reprobados")
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = df['Aprobado'].value_counts()
        colors = ['salmon', 'lightblue']
        ax.bar(['Reprobados', 'Aprobados'], counts.values, color=colors, edgecolor='black', width=0.6)
        ax.set_ylabel('Cantidad de Estudiantes')
        ax.set_title('Distribución de Estudiantes Aprobados/Reprobados', fontsize=14, fontweight='bold')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 100, str(v), ha='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("Relación entre Variables")
        
        # Scatter plot: Asistencia vs Nota Final
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = df.sample(n=min(3000, len(df)), random_state=42)
        scatter = ax.scatter(sample['Asistencia'], sample['Nota_final'], 
                           c=sample['Aprobado'], cmap='RdYlGn', 
                           alpha=0.5, edgecolors='black', s=30)
        ax.set_xlabel('Asistencia (%)', fontsize=12)
        ax.set_ylabel('Nota Final', fontsize=12)
        ax.set_title('Asistencia vs Nota Final (Muestra)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Aprobado (1=Sí, 0=No)')
        st.pyplot(fig)
        plt.close()
        
        # Correlación
        corr = df[['Asistencia', 'Nota_final', 'Aprobado']].corr()
        st.subheader("Matriz de Correlación")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlación entre Variables', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ============================================================
# PÁGINA: MODELO SUPERVISADO
# ============================================================

elif page == "🎯 Modelo Supervisado":
    st.header("🎯 Modelo Supervisado - Clasificación")
    st.markdown("**Predicción de Aprobación usando Regresión Logística**")
    
    # Preparar datos
    X, y, features = prepare_supervised_data(df)
    
    st.sidebar.subheader("⚙️ Configuración del Modelo")
    test_size = st.sidebar.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.3, 0.05)
    random_state = st.sidebar.number_input("Semilla aleatoria", 1, 100, 42)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Mostrar información de los datos
    st.subheader("📊 Información del Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de datos", len(X))
    col2.metric("Entrenamiento", len(X_train))
    col3.metric("Prueba", len(X_test))
    col4.metric("Features", len(features))
    
    st.write(f"**Features seleccionados:** {', '.join(features)}")
    
    # Entrenar modelo
    if st.button("🚀 Entrenar Modelo de Regresión Logística", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            # Entrenar
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Métricas
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            st.success("✅ Modelo entrenado exitosamente!")
            
            # Mostrar resultados
            st.subheader("📈 Resultados del Modelo")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Accuracy (Entrenamiento)", f"{train_accuracy:.2%}")
            col2.metric("🎯 Accuracy (Prueba)", f"{test_accuracy:.2%}")
            col3.metric("📊 Diferencia", f"{abs(train_accuracy - test_accuracy):.2%}")
            
            # Matriz de Confusión
            st.subheader("🔲 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred_test)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Reprobado', 'Aprobado'],
                       yticklabels=['Reprobado', 'Aprobado'],
                       cbar_kws={'label': 'Cantidad'},
                       linewidths=2, linecolor='black', ax=ax)
            ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
            ax.set_title('Matriz de Confusión - Modelo Supervisado', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
            # Interpretación de la matriz
            tn, fp, fn, tp = cm.ravel()
            st.write(f"""
            **Interpretación de la Matriz de Confusión:**
            - ✅ **Verdaderos Negativos (TN)**: {tn} - Correctamente predijo reprobados
            - ❌ **Falsos Positivos (FP)**: {fp} - Predijo aprobado pero era reprobado
            - ❌ **Falsos Negativos (FN)**: {fn} - Predijo reprobado pero era aprobado
            - ✅ **Verdaderos Positivos (TP)**: {tp} - Correctamente predijo aprobados
            """)
            
            # Reporte de Clasificación
            st.subheader("📋 Reporte de Clasificación Detallado")
            report = classification_report(y_test, y_pred_test, 
                                          target_names=['Reprobado', 'Aprobado'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            
            # Importancia de features
            if hasattr(model, 'coef_'):
                st.subheader("📊 Importancia de Variables")
                
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Coeficiente': model.coef_[0],
                    'Importancia': np.abs(model.coef_[0])
                }).sort_values('Importancia', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['green' if x > 0 else 'red' for x in feature_importance['Coeficiente']]
                ax.barh(feature_importance['Feature'], feature_importance['Importancia'], 
                       color=colors, edgecolor='black')
                ax.set_xlabel('Importancia Absoluta', fontsize=12)
                ax.set_title('Importancia de Variables en la Predicción', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                plt.close()
                
                st.dataframe(feature_importance, use_container_width=True, hide_index=True)
            
            # Interpretación final
            st.markdown("---")
            st.subheader("💡 Interpretación de Resultados")
            
            if test_accuracy >= 0.85:
                st.success(f"""
                ✅ **Excelente rendimiento del modelo ({test_accuracy:.2%})**
                - El modelo es muy confiable para predecir aprobación/reprobación
                - Alta precisión en ambas clases
                """)
            elif test_accuracy >= 0.75:
                st.info(f"""
                👍 **Buen rendimiento del modelo ({test_accuracy:.2%})**
                - El modelo es útil para predicciones
                - Se puede mejorar con más features
                """)
            else:
                st.warning(f"""
                ⚠️ **Rendimiento moderado del modelo ({test_accuracy:.2%})**
                - Considerar agregar más variables predictoras
                - Evaluar otros algoritmos
                """)
            
            st.write("""
            **Conclusiones del Modelo Supervisado:**
            - Tipo: **Aprendizaje Supervisado (Clasificación)**
            - Algoritmo: **Regresión Logística**
            - Objetivo: Predecir si un estudiante aprobará o reprobará
            - Variables más influyentes: {}
            """.format(', '.join(feature_importance.head(2)['Feature'].tolist())))

# ============================================================
# PÁGINA: MODELO NO SUPERVISADO
# ============================================================

elif page == "🔍 Modelo No Supervisado":
    st.header("🔍 Modelo No Supervisado - Clustering")
    st.markdown("**Agrupación de estudiantes usando K-Means**")
    
    # Preparar datos
    df_cluster = prepare_clustering_data(df)
    
    st.sidebar.subheader("⚙️ Configuración de K-Means")
    n_clusters = st.sidebar.slider("Número de clusters (k)", 2, 5, 3)
    random_state = st.sidebar.number_input("Semilla aleatoria", 1, 100, 42)
    
    st.subheader("📊 Información del Dataset para Clustering")
    col1, col2 = st.columns(2)
    col1.metric("Registros válidos", len(df_cluster))
    col2.metric("Features", "Asistencia, Nota_final")
    
    # Mostrar muestra de datos
    st.write("**Muestra de datos para clustering:**")
    st.dataframe(df_cluster.head(10), use_container_width=True)
    
    if st.button("🔍 Aplicar K-Means Clustering", type="primary", use_container_width=True):
        with st.spinner("Aplicando clustering..."):
            # Escalar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            
            # Aplicar K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Agregar clusters al dataframe
            df_cluster['Cluster'] = clusters
            
            st.success(f"✅ Clustering completado con {n_clusters} grupos!")
            
            # Visualización principal
            st.subheader("📊 Visualización de Clusters")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Graficar cada cluster con diferente color
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
            for i in range(n_clusters):
                cluster_data = df_cluster[df_cluster['Cluster'] == i]
                ax.scatter(cluster_data['Asistencia'], cluster_data['Nota_final'],
                          c=[colors[i]], label=f'Cluster {i}', 
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
            
            # Graficar centroides
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(centroids[:, 0], centroids[:, 1],
                      c='red', s=500, alpha=0.9, marker='X',
                      edgecolors='black', linewidths=3, label='Centroides')
            
            ax.set_xlabel('Asistencia (%)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Nota Final', fontsize=14, fontweight='bold')
            ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12, loc='best')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Estadísticas por cluster
            st.subheader("📊 Estadísticas por Cluster")
            
            cluster_stats = df_cluster.groupby('Cluster').agg({
                'Asistencia': ['mean', 'std', 'min', 'max', 'count'],
                'Nota_final': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
            cluster_stats = cluster_stats.reset_index()
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Distribución de clusters
            st.subheader("📈 Distribución de Estudiantes por Cluster")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
            bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                         color=colors[:n_clusters], edgecolor='black', width=0.6)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}\n({height/len(df_cluster)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cantidad de Estudiantes', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Estudiantes por Cluster', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()
            
            # Interpretación de clusters
            st.markdown("---")
            st.subheader("💡 Interpretación de Clusters")
            
            for i in range(n_clusters):
                cluster_data = df_cluster[df_cluster['Cluster'] == i]
                avg_asistencia = cluster_data['Asistencia'].mean()
                avg_nota = cluster_data['Nota_final'].mean()
                count = len(cluster_data)
                
                # Determinar perfil
                if avg_nota >= 8 and avg_asistencia >= 85:
                    perfil = "🌟 **Estudiantes Exitosos**"
                    descripcion = "Alta asistencia y excelentes notas. Son estudiantes modelo."
                    color = "success"
                elif avg_nota < 7 and avg_asistencia < 70:
                    perfil = "⚠️ **Estudiantes en Riesgo**"
                    descripcion = "Baja asistencia y notas bajas. **Requieren intervención urgente.**"
                    color = "error"
                elif avg_asistencia >= 80 and avg_nota < 7.5:
                    perfil = "🤔 **Necesitan Apoyo Académico**"
                    descripcion = "Buena asistencia pero dificultades académicas. Necesitan tutorías."
                    color = "warning"
                else:
                    perfil = "📈 **Rendimiento Medio**"
                    descripcion = "Rendimiento aceptable con margen de mejora."
                    color = "info"
                
                # Mostrar análisis del cluster
                with st.expander(f"**Cluster {i}** - {count} estudiantes ({count/len(df_cluster)*100:.1f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("👥 Cantidad", count)
                        st.metric("📊 Asistencia Promedio", f"{avg_asistencia:.1f}%")
                        st.metric("📝 Nota Promedio", f"{avg_nota:.2f}")
                    
                    with col2:
                        if color == "success":
                            st.success(f"""
                            {perfil}
                            
                            {descripcion}
                            
                            **Características:**
                            - Asistencia: {avg_asistencia:.1f}%
                            - Nota: {avg_nota:.2f}
                            """)
                        elif color == "error":
                            st.error(f"""
                            {perfil}
                            
                            {descripcion}
                            
                            **Características:**
                            - Asistencia: {avg_asistencia:.1f}%
                            - Nota: {avg_nota:.2f}
                            """)
                        elif color == "warning":
                            st.warning(f"""
                            {perfil}
                            
                            {descripcion}
                            
                            **Características:**
                            - Asistencia: {avg_asistencia:.1f}%
                            - Nota: {avg_nota:.2f}
                            """)
                        else:
                            st.info(f"""
                            {perfil}
                            
                            {descripcion}
                            
                            **Características:**
                            - Asistencia: {avg_asistencia:.1f}%
                            - Nota: {avg_nota:.2f}
                            """)
            
            # Conclusiones
            st.markdown("---")
            st.subheader("📝 Conclusiones del Clustering")
            st.write("""
            **Tipo de Modelo:** Aprendizaje No Supervisado (Clustering)
            
            **Algoritmo:** K-Means
            
            **Objetivo:** Agrupar estudiantes con patrones similares de rendimiento
            
            **Hallazgos clave:**
            - Se identificaron {} grupos distintos de estudiantes
            - Los clusters revelan patrones claros de rendimiento académico
            - La asistencia es un factor diferenciador importante entre grupos
            - Permite personalizar estrategias de apoyo por perfil de estudiante
            """.format(n_clusters))
