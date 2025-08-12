import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
import seaborn as sns
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
import time
from collections import defaultdict
import json

class ModelEvaluator:
    def __init__(self):
        self.setup_environment()
        self.classes = None
        self.results = {}
        
    def setup_environment(self):
        """Configura el entorno visual y de directorios"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        os.makedirs('results/individual', exist_ok=True)
        os.makedirs('results/combined', exist_ok=True)
        
    def load_data(self):
        """Carga todos los datos necesarios"""
        # Cargar datos reales
        self.y_true = np.load("y_true.npy")
        self.classes = np.unique(self.y_true)
        
        # Cargar predicciones combinadas
        self.combined_pred = np.load("combined_predictions.npy")
        self.combined_proba = np.load("combined_predictions_proba.npy")
        
        # Cargar predicciones individuales de CNN
        cnn_data = np.load("cnn_predictions.npy", allow_pickle=True).item()
        self.cnn_pred = np.argmax(cnn_data['predictions'], axis=1)
        self.cnn_proba = cnn_data['predictions']
        
        # Cargar predicciones individuales de ExtraTrees
        self.et_proba = np.load("extratrees_predictions_proba.npy")
        self.et_pred = np.argmax(self.et_proba, axis=1)
        
    def evaluate_model(self, y_pred, probas, model_name):
        """Evalúa un modelo individual"""
        # Métricas básicas
        metrics = {
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_true, y_pred, average='weighted', zero_division=0),
            'report': classification_report(self.y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        # Métricas ROC y PR
        y_true_bin = label_binarize(self.y_true, classes=self.classes)
        roc_metrics = {}
        pr_metrics = {}
        
        for i in range(len(self.classes)):
            # ROC
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probas[:, i])
            roc_metrics[f'class_{i}'] = {
                'auc': auc(fpr, tpr),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probas[:, i])
            pr_metrics[f'class_{i}'] = {
                'ap': average_precision_score(y_true_bin[:, i], probas[:, i]),
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }
        
        metrics.update({
            'roc_metrics': roc_metrics,
            'pr_metrics': pr_metrics
        })
        
        return metrics
    
    def generate_visualizations(self, metrics, model_name):
        """Genera visualizaciones para un modelo específico"""
        # Directorio específico para el modelo
        model_dir = f'results/{model_name.replace(" ", "_").lower()}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_true, metrics['y_pred'])
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{model_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        
        # Curva ROC
        plt.figure()
        for i, cls in enumerate(self.classes):
            roc_data = metrics['roc_metrics'][f'class_{i}']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'{cls} (AUC = {roc_data["auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_dir}/roc_curve.png', dpi=300)
        plt.close()
        
        # Distribución de probabilidades
        plt.figure()
        for i, cls in enumerate(self.classes):
            sns.kdeplot(metrics['probas'][self.y_true == i][:, i], 
                       label=f'Clase {cls}', fill=True)
        plt.title(f'Distribución de Probabilidades - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_dir}/probability_distribution.png', dpi=300)
        plt.close()
    
    def save_results(self, metrics, model_name):
        """Guarda los resultados de cada modelo"""
        model_dir = f'results/{model_name.replace(" ", "_").lower()}'
        
        # Guardar métricas básicas
        with open(f'{model_dir}/metrics.json', 'w') as f:
            json.dump({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }, f, indent=4)
        
        # Guardar reporte de clasificación
        pd.DataFrame(metrics['report']).transpose().to_csv(f'{model_dir}/classification_report.csv')
    
    def compare_models(self):
        """Compara el rendimiento de todos los modelos"""
        comparison = pd.DataFrame({
            'CNN': [
                self.results['CNN']['accuracy'],
                self.results['CNN']['precision'],
                self.results['CNN']['recall'],
                self.results['CNN']['f1']
            ],
            'ExtraTrees': [
                self.results['ExtraTrees']['accuracy'],
                self.results['ExtraTrees']['precision'],
                self.results['ExtraTrees']['recall'],
                self.results['ExtraTrees']['f1']
            ],
            'Combinado': [
                self.results['Combinado']['accuracy'],
                self.results['Combinado']['precision'],
                self.results['Combinado']['recall'],
                self.results['Combinado']['f1']
            ]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        # Guardar comparación
        os.makedirs('results/comparison', exist_ok=True)
        comparison.to_csv('results/comparison/model_comparison.csv')
        comparison.to_markdown('results/comparison/model_comparison.md')
        
        # Gráfico de comparación
        plt.figure()
        comparison.T.plot(kind='bar', figsize=(12, 6))
        plt.title('Comparación de Modelos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/comparison/model_comparison.png', dpi=300)
        plt.close()
        
        return comparison
    
    def run_evaluation(self):
        """Ejecuta toda la evaluación"""
        start_time = time.time()
        self.load_data()
        
        # Evaluar modelos individuales y combinado
        for name, pred, proba in [
            ('CNN', self.cnn_pred, self.cnn_proba),
            ('ExtraTrees', self.et_pred, self.et_proba),
            ('Combinado', self.combined_pred, self.combined_proba)
        ]:
            print(f"\nEvaluando modelo: {name}")
            self.results[name] = self.evaluate_model(pred, proba, name)
            self.results[name].update({
                'y_pred': pred,
                'probas': proba
            })
            self.generate_visualizations(self.results[name], name)
            self.save_results(self.results[name], name)
        
        # Comparar modelos
        comparison = self.compare_models()
        
        # Resumen final
        print("\n" + "="*50)
        print("Resumen de Evaluación".center(50))
        print("="*50)
        print(comparison)
        print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f} segundos")
        
        # Guardar resumen ejecutivo
        with open('results/comparison/executive_summary.txt', 'w') as f:
            f.write("RESUMEN EJECUTIVO DE MODELOS\n")
            f.write("="*50 + "\n")
            f.write(comparison.to_string())
            f.write("\n\nRecomendación: ")
            
            best_model = comparison.loc['F1-Score'].idxmax()
            improvement = (comparison.loc['F1-Score'][best_model] - 
                          comparison.loc['F1-Score']['Combinado']) / comparison.loc['F1-Score']['Combinado'] * 100
            
            if best_model == 'Combinado':
                f.write("La combinación de modelos es la mejor estrategia")
            else:
                f.write(f"El modelo individual {best_model} supera a la combinación por {improvement:.2f}% en F1-Score")

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()