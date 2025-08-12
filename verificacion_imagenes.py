import os
import pandas as pd

# Cargar el DataFrame metadata_df desde un archivo CSV (ajusta la ruta según corresponda)
metadata_df = pd.read_csv(r'data\metadata.csv')  # Ruta relativa desde donde ejecutas el script

# Extraer solo el nombre base sin extensión en metadata_df
metadata_df['img_id_base'] = metadata_df['img_id'].str.replace(r'\.png$', '', regex=True).str.replace(r'\.jpg$', '', regex=True)

# Define la ruta al directorio de imágenes
IMAGE_DIR = 'data\imagenes'  # Cambia esto por la ruta real de tu carpeta de imágenes

# Lista de archivos reales (sin extensión)
archivos = os.listdir(IMAGE_DIR)
img_ids_en_carpeta = set(os.path.splitext(f)[0] for f in archivos if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg'])

# img_ids en metadata base (sin extensión)
img_ids_metadata = set(metadata_df['img_id_base'])

# Imágenes en metadata pero NO en carpeta
img_ids_faltantes = img_ids_metadata - img_ids_en_carpeta
print(f"Cantidad de imágenes en metadata pero que NO existen en la carpeta: {len(img_ids_faltantes)}")

# Imágenes en carpeta pero NO en metadata
img_ids_extra = img_ids_en_carpeta - img_ids_metadata
print(f"Cantidad de imágenes en carpeta pero que NO están en metadata: {len(img_ids_extra)}")

# Imágenes válidas que sí están en ambos
img_ids_validos = img_ids_metadata & img_ids_en_carpeta
print(f"Cantidad de imágenes válidas para cargar: {len(img_ids_validos)}")
