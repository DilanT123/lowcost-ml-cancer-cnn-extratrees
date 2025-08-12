# reparar_imagenes.py
import os
from PIL import Image
from io import BytesIO
import numpy as np

def reparar_imagen(img_path):
    try:
        # Intenta leer la imagen como binario primero
        with open(img_path, 'rb') as f:
            img_data = f.read()
        
        # Intenta abrir con PIL
        img = Image.open(BytesIO(img_data))
        img.verify()
        img = img.convert('RGB')
        
        # Si pasa la verificación, guardar nuevamente
        img.save(img_path, quality=95)
        return True
    except Exception as e:
        print(f"No se pudo reparar {img_path}: {str(e)}")
        return False

# Ejecutar esto para todas las imágenes problemáticas
image_dir = 'data/imagenes/'
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, filename)
        try:
            with Image.open(img_path) as img:
                img.verify()
        except:
            print(f"Reparando {filename}...")
            reparar_imagen(img_path)