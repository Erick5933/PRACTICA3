import pandas as pd
import os

# Directorio donde se encuentran los archivos
data_dir = "academic_data"
files = [f for f in os.listdir(data_dir) if f.endswith('.xls')]
files.sort()

# Usaremos el primer archivo para la inspección inicial
first_file = os.path.join(data_dir, files[0])

print(f"Inspeccionando el archivo: {first_file}")

try:
    # Intentar leer el archivo .xls. El motor 'xlrd' es necesario para archivos .xls antiguos.
    # Leer solo las primeras filas para una inspección rápida
    df = pd.read_excel(first_file, engine='xlrd', nrows=5)
    
    print("\nPrimeras 5 filas del DataFrame:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
    
    print("\nInformación de las columnas (dtype y no-nulos):")
    df.info()
    
    # Guardar la estructura de las columnas para referencia
    with open("data_structure.txt", "w") as f:
        f.write(df.head().to_markdown(index=False, numalign="left", stralign="left"))
        f.write("\n\nColumn Info:\n")
        df.info(buf=f)

except Exception as e:
    print(f"Error al leer el archivo {first_file}: {e}")

# Mostrar la lista de todos los archivos
print("\nLista de todos los archivos encontrados:")
for f in files:
    print(f)
