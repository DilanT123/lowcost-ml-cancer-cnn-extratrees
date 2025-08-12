import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score, classification_report

def load_predictions(filename, expected_samples, expected_classes):
    try:
        data = np.load(filename)
        if data.shape != (expected_samples, expected_classes):
            raise ValueError(f"Dimensiones incorrectas en {filename}")
        return data
    except Exception as e:
        print(f"Error cargando {filename}: {str(e)}")
        sys.exit(1)

def main():
    try:
        # Crear directorio de resultados si no existe
        os.makedirs('results', exist_ok=True)
        
        # Cargar predicciones de CNN
        cnn_data = np.load("cnn_predictions.npy", allow_pickle=True).item()
        cnn_proba = cnn_data['predictions']
        y_test = cnn_data['true_labels']  # Usamos las etiquetas de CNN como referencia
        
        # Verificar dimensiones
        num_samples, num_classes = cnn_proba.shape
        
        # Cargar predicciones de ExtraTrees
        et_proba = load_predictions(
            "extratrees_predictions_proba.npy", 
            num_samples, 
            num_classes
        )
        
        # Verificar consistencia de etiquetas
        y_test_et = np.load("y_test_extratrees.npy")
        if not np.array_equal(y_test, y_test_et):
            print("Advertencia: Las etiquetas de test no coinciden entre CNN y ExtraTrees")
            print("Usando las etiquetas de CNN como referencia")
        
        # Fusión ponderada (70% CNN + 30% ExtraTrees)
        combined_proba = 0.7 * cnn_proba + 0.3 * et_proba
        final_predictions = np.argmax(combined_proba, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, final_predictions)
        report = classification_report(y_test, final_predictions, zero_division=0)
        
        # Guardar resultados
        np.save("results/combined_predictions.npy", final_predictions)
        np.save("results/combined_probabilities.npy", combined_proba)
        np.save("results/y_true.npy", y_test)
        
        # Guardar reporte en texto
        with open("results/fusion_report.txt", "w") as f:
            f.write(f"Late Fusion Results\n{'='*40}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print("\nLate Fusion completada exitosamente!")
        print(f"Accuracy combinado: {accuracy:.4f}")
        print("\nResultados guardados en la carpeta 'results'")
        print(report)
        
    except Exception as e:
        print(f"\nError en la fusión: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()