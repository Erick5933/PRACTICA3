import pandas as pd
import os
from collections import Counter

# -----------------------------
# Configuración
# -----------------------------
data_dir = "academic_data"
# columnas normalizadas
new_column_names = [
    "Periodo", "Paralelo", "Identificacion_Estudiante", "Estudiante", "Carrera",
    "Nivel", "Asignatura", "Num_matricula", "Asistencia", "Nota_final",
    "Estado_Asignatura", "Estado_Matricula", "Tipo_Ingreso", "Cedula_docente",
    "Nombre_docente"
]
# filtros de carreras
ban_list = {'PPTC-V-PREPARATEC-VIRTUAL', 'CDI-CENTRO DE IDIOMAS', 'CDI-N-CENTRO DE IDIOMAS NUEVA'}

# -----------------------------
# Recolectar archivos
# -----------------------------
files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.xls', '.xlsx'))]
files.sort()
print(f"Iniciando la carga y procesamiento de {len(files)} archivos...")

all_data = []

# -----------------------------
# Función de lectura robusta
# -----------------------------
def read_excel_segment(path):
    """
    Lee un archivo Excel con dos pasadas:
    1) sin encabezado para ubicar índices de columnas
    2) luego recorta desde fila 4 (índice 3) y asigna nombres
    Devuelve df con columnas estandarizadas.
    """
    # Elegir engine
    engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"

    # Leer crudo sin encabezado
    df_raw = pd.read_excel(path, header=None, engine=engine)

    # Seleccionar filas de datos (desde la 4ta fila) y columnas por índice 0-based
    # Índices: 1,2,3,4,5,6,7,8,10,11,12,14,15,16,17
    col_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17]
    df_clean = df_raw.iloc[3:, col_idx].copy()
    df_clean.columns = new_column_names

    # Cast básicos
    df_clean['Nota_final'] = pd.to_numeric(df_clean['Nota_final'], errors='coerce')
    # eliminar filas sin nota_final (encabezados/pies)
    df_clean.dropna(subset=['Nota_final'], inplace=True)

    return df_clean

# -----------------------------
# Carga y consolidación
# -----------------------------
for file_name in files:
    file_path = os.path.join(data_dir, file_name)
    try:
        df_clean = read_excel_segment(file_path)
        all_data.append(df_clean)
        print(f"Archivo {file_name} cargado con {len(df_clean)} registros.")
    except Exception as e:
        print(f"Error al procesar el archivo {file_name}: {e}")

if not all_data:
    print("⚠️ No se pudo cargar ningún archivo. Revisa el directorio o el formato.")
else:
    # Concatenar
    df_master = pd.concat(all_data, ignore_index=True)

    # Guardar versión cruda consolidada (antes de filtros)
    raw_path = "academic_performance_master_raw.csv"
    df_master.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"\n✔️ Datos consolidados (sin filtrar) en: {raw_path}")
    print(f"Total de registros consolidados: {len(df_master)}")

    # -----------------------------
    # Aplicar filtros de carreras
    # -----------------------------
    if 'Carrera' not in df_master.columns:
        raise ValueError("La columna 'Carrera' no existe en el DataFrame consolidado.")

    # Normalizar para comparación
    df_master['Carrera'] = df_master['Carrera'].astype(str).str.strip()
    carrera_upper = df_master['Carrera'].str.upper()

    total_antes = len(df_master)

    # Conteo de removidos por criterio (para log)
    removal_reasons = Counter()

    # Marcas de filtrado
    mask_ban_list = carrera_upper.isin({s.upper() for s in ban_list})
    mask_nt = carrera_upper.str.startswith('NT-')

    removal_reasons['ban_list'] = int(mask_ban_list.sum())
    removal_reasons['NT-*'] = int(mask_nt.sum())

    # Aplicar filtro combinado
    df_filtered = df_master[~(mask_ban_list | mask_nt)].copy()

    total_despues = len(df_filtered)
    total_removidos = total_antes - total_despues

    # Guardar versión filtrada
    filtered_path = "academic_performance_master.csv"
    df_filtered.to_csv(filtered_path, index=False, encoding="utf-8")
    print(f"✔️ Datos filtrados guardados en: {filtered_path}")
    print(f"Total después de filtrar: {total_despues} (removidos: {total_removidos})")

    # Mostrar primeras filas en consola (sin tabulate, para evitar dependencias)
    print("\nPrimeras 5 filas del DataFrame filtrado:")
    print(df_filtered.head().to_string(index=False))

    # -----------------------------
    # Guardar resumen de estructura
    # -----------------------------
    with open("master_data_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total de registros consolidados (raw): {total_antes}\n")
        f.write(f"Removidos por ban_list: {removal_reasons['ban_list']}\n")
        f.write(f"Removidos por NT-*   : {removal_reasons['NT-*']}\n")
        f.write(f"Total después de filtrar: {total_despues}\n")
        f.write("\nColumnas del DataFrame filtrado:\n")
        for c, t in zip(df_filtered.columns, df_filtered.dtypes):
            f.write(f" - {c}: {t}\n")
        f.write("\nPrimeras 5 filas (to_string):\n")
        f.write(df_filtered.head().to_string(index=False))
