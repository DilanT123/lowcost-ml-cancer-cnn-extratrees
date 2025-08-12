import sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import pandas as pd
import joblib
import time
import os

def run_extratrees_training():
    try:
        # Configuración inicial
        os.makedirs('models', exist_ok=True)
        
        # Cargar datos
        clinical_data = pd.read_csv('processed_metadata.csv')
        train_idx = np.load('train_indices.npy')
        test_idx = np.load('test_indices.npy')
        
        # Preparar datos
        X = clinical_data.drop(['img_id', 'diagnostic_numeric'], axis=1)
        y = clinical_data['diagnostic_numeric']
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Normalización
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configuración de Extra Trees optimizada
        print("\n--- Entrenando ExtraTreesClassifier ---")
        
        # Espacio de parámetros mejorado
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(np.arange(10, 50, 5)),
            'min_samples_split': randint(2, 15),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample'],
            'criterion': ['gini', 'entropy']
        }
        
        # Modelo base con buenos parámetros por defecto
        et_model = ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )
        
        # Búsqueda optimizada
        search = RandomizedSearchCV(
            estimator=et_model,
            param_distributions=param_dist,
            n_iter=50,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            verbose=2,
            n_jobs=1,  # ExtraTrees ya usa paralelismo interno
            random_state=42
        )
        
        # Entrenamiento con temporizador
        start_time = time.time()
        print("\nIniciando búsqueda de hiperparámetros...")
        search.fit(X_train_scaled, y_train)
        print(f"\nBúsqueda completada en {(time.time()-start_time)/60:.2f} minutos")
        
        # Mejor modelo
        best_et = search.best_estimator_
        print("\nMejores parámetros encontrados:")
        print(search.best_params_)
        print(f"Mejor puntaje de validación: {search.best_score_:.4f}")
        
        # Evaluación en test
        test_acc = best_et.score(X_test_scaled, y_test)
        print(f"\nAccuracy en conjunto de prueba: {test_acc:.4f}")
        
        # Guardar modelo y recursos
        joblib.dump(best_et, 'models/best_extratrees_model.pkl')
        joblib.dump(scaler, 'models/extratrees_scaler.pkl')
        np.save('models/extratrees_feature_importances.npy', best_et.feature_importances_)
        
        # Guardar predicciones
        probas = best_et.predict_proba(X_test_scaled)
        np.save('extratrees_predictions_proba.npy', probas)
        np.save('y_test_extratrees.npy', y_test)
        
        print("\nRecursos guardados:")
        print("- models/best_extratrees_model.pkl (modelo entrenado)")
        print("- models/extratrees_scaler.pkl (normalizador)")
        print("- models/extratrees_feature_importances.npy (importancias de características)")
        print("- extratrees_predictions_proba.npy (predicciones)")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_extratrees_training()