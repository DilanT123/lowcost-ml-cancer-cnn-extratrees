import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

def load_and_preprocess_data():
    """Carga y limpieza inicial de datos"""
    metadata = pd.read_csv("data/metadata.csv")
    metadata = metadata.drop(columns=["lesion_id", "Unnamed: 0"], errors="ignore")
    metadata.fillna({'age': metadata['age'].median()}, inplace=True)
    return metadata

def preprocess_features(df):
    """Preprocesamiento de características"""
    img_ids = df["img_id"].copy()
    numeric_features = df.select_dtypes(include=np.number).columns.drop('img_id', errors='ignore')
    categorical_features = df.select_dtypes(include="object").columns.drop(['img_id', 'diagnostic'], errors='ignore')
    return img_ids, numeric_features, categorical_features

def encode_target(y):
    """Codificación de la variable objetivo"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le.classes_

def apply_smote_balance(X, y, img_ids):
    """Aplicar SMOTE manteniendo los img_ids correspondientes"""
    print("\nDistribución de clases antes de balanceo:", Counter(y))
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Generar img_ids sintéticos para las nuevas muestras
    new_samples = len(X_res) - len(X)
    synthetic_ids = [f"SYNTH_{i}" for i in range(new_samples)]
    img_ids_res = np.concatenate([img_ids.values, synthetic_ids])
    
    print("Distribución de clases después de balanceo:", Counter(y_res))
    return X_res, y_res, img_ids_res

def main():
    print("=== Procesamiento de Datos ===")
    
    # 1. Carga y limpieza
    metadata = load_and_preprocess_data()
    
    # 2. Separar características y objetivo
    X = metadata.drop(columns=["diagnostic"], errors="ignore")
    y = metadata["diagnostic"]
    
    # 3. Codificar objetivo
    y_encoded, class_names = encode_target(y)
    print("\nClases codificadas:", dict(zip(class_names, range(len(class_names)))))
    
    # 4. Preprocesar características
    img_ids, numeric_features, categorical_features = preprocess_features(X)
    
    # Imputación numérica
    num_imputer = SimpleImputer(strategy="median")
    X_numeric = num_imputer.fit_transform(X[numeric_features])
    
    # Codificación categórica
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_categorical = cat_encoder.fit_transform(X[categorical_features])
    
    # Combinar características
    X_processed = np.concatenate([X_numeric, X_categorical], axis=1)
    
    # 5. Balanceo de clases (conservando img_ids)
    X_resampled, y_resampled, img_ids_resampled = apply_smote_balance(
        X_processed, y_encoded, img_ids
    )
    
    # 6. Dividir datos (índices estratificados)
    indices = np.arange(len(X_resampled))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )
    
    # 7. Crear DataFrame final
    processed_data = pd.DataFrame(X_resampled)
    processed_data['img_id'] = img_ids_resampled
    processed_data['diagnostic_numeric'] = y_resampled
    
    # 8. Guardar resultados
    os.makedirs('processed', exist_ok=True)
    processed_data.to_csv("processed/processed_metadata.csv", index=False)
    np.save('processed/train_indices.npy', train_idx)
    np.save('processed/test_indices.npy', test_idx)
    
    # 9. Resumen
    print("\n=== Resumen Final ===")
    print(f"Muestras totales: {len(processed_data)}")
    print(f"Distribución de clases final:")
    for cls, count in Counter(y_resampled).items():
        print(f"- {class_names[cls]}: {count} muestras")
    print("\nArchivos guardados en directorio 'processed/'")

if __name__ == '__main__':
    main()