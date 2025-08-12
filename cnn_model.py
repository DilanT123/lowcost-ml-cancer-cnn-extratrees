from pyexpat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.utils import class_weight
import time
import warnings
warnings.filterwarnings('ignore')

# --- Configuración optimizada ---
IMAGE_DIR = 'data/imagenes/'
CNN_PREDICTIONS_PATH = 'cnn_predictions.npy'
BATCH_SIZE = 32  # Reducido para manejar posibles problemas de memoria
NUM_WORKERS = 2   # Reducido para mayor estabilidad
PREFETCH_FACTOR = 1  # Reducido para problemas de memoria

# --- Transformaciones optimizadas ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- Dataset ultra-robusto ---
class RobustSkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = self._load_valid_samples()
        
        # Reportar estadísticas
        self._report_stats()
    
    def _load_valid_samples(self):
        valid_samples = []
        for _, row in self.df.iterrows():
            img_path = self._find_valid_image_path(row['img_id_base'])
            if img_path:
                valid_samples.append((img_path, row['diagnostic_numeric']))
        return valid_samples
    
    def _find_valid_image_path(self, img_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                try:
                    # Doble verificación de la imagen
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        img.convert('RGB')  # Verificar conversión
                    return img_path
                except Exception as e:
                    print(f"Imagen inválida (será omitida): {img_path} - Error: {str(e)}")
        return None
    
    def _report_stats(self):
        print(f"\n📊 Estadísticas del Dataset:")
        print(f"- Registros totales en metadata: {len(self.df)}")
        print(f"- Muestras válidas encontradas: {len(self.samples)}")
        print(f"- Porcentaje válido: {len(self.samples)/len(self.df):.2%}")
        
        if len(self.samples) > 0:
            class_counts = pd.Series([label for _, label in self.samples]).value_counts()
            print("\n📈 Distribución de clases (válidas):")
            print(class_counts.to_string())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            print(f"⚠️ Error cargando {img_path}: {str(e)} - Usando imagen dummy")
            return torch.zeros(3, 64, 64), label

# --- Arquitectura CNN Optimizada ---
class EfficientCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Función principal con manejo completo de errores ---
def run_cnn_training():
    try:
        # Configuración CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        print(f"\n🔧 Configuración:")
        print(f"- Dispositivo: {device}")
        print(f"- Versión CUDA: {torch.version.cuda if device.type == 'cuda' else 'No disponible'}")
        
        # 1. Cargar metadatos
        print("\n📂 Cargando metadatos...")
        metadata_df = pd.read_csv('processed_metadata.csv')
        metadata_df['img_id_base'] = metadata_df['img_id'].str.replace(r'\.(png|jpg|jpeg)$', '', regex=True)
        metadata_df['diagnostic_numeric'] = pd.to_numeric(metadata_df['diagnostic_numeric'])
        metadata_df = metadata_df.dropna(subset=['diagnostic_numeric'])
        metadata_df = metadata_df.reset_index(drop=True)
        
        # Verificar clases presentes
        unique_classes = np.unique(metadata_df['diagnostic_numeric'])
        print(f"- Clases encontradas: {unique_classes}")
        print(f"- Total registros: {len(metadata_df)}")

        # 2. Cargar índices de train/test
        print("\n🔍 Cargando división train/test...")
        train_idx = np.load('train_indices.npy')
        test_idx = np.load('test_indices.npy')
        
        train_df = metadata_df.iloc[train_idx].reset_index(drop=True)
        test_df = metadata_df.iloc[test_idx].reset_index(drop=True)
        
        # 3. Crear datasets
        print("\n🖼️ Preparando datasets...")
        transform = get_transforms()
        train_dataset = RobustSkinLesionDataset(train_df, IMAGE_DIR, transform)
        test_dataset = RobustSkinLesionDataset(test_df, IMAGE_DIR, transform)
        
        # Verificar que tenemos muestras suficientes
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError("No hay suficientes imágenes válidas para entrenamiento o prueba")
        
        # 4. Balanceo de clases con verificación robusta
        print("\n⚖️ Balanceando clases...")
        y_train = np.array([label for _, label in train_dataset.samples])
        unique_classes = np.unique(y_train)
        
        # Verificar que tenemos muestras de todas las clases esperadas
        expected_classes = np.unique(metadata_df['diagnostic_numeric'])
        if len(unique_classes) != len(expected_classes):
            missing_classes = set(expected_classes) - set(unique_classes)
            print(f"⚠️ Advertencia: Faltan muestras para las clases: {missing_classes}")
            
            # Calcular pesos usando todos los datos disponibles
            full_class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=expected_classes,
                y=metadata_df['diagnostic_numeric']
            )
            class_weights = full_class_weights[np.isin(expected_classes, unique_classes)]
        else:
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train
            )
        
        # Convertir a tensor
        sample_weights = torch.tensor([class_weights[cls] for cls in y_train], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # 5. DataLoaders
        print("\n🔄 Configurando DataLoaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR
        )

        # 6. Modelo
        print("\n🧠 Inicializando modelo...")
        model = EfficientCNN(num_classes=len(expected_classes)).to(device)
        
        # Loss y optimizador
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        # 7. Entrenamiento
        print("\n🚀 Comenzando entrenamiento...")
        best_loss = float('inf')
        
        for epoch in range(15):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Log de progreso cada 10 batches
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Validación
            model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(targets.cpu().numpy())
            
            scheduler.step(val_loss)
            
            # Guardar mejor modelo
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_cnn_model.pth')
                print(f"💾 Modelo guardado (val_loss: {val_loss:.4f})")
            
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{15} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(test_loader):.4f} | "
                  f"Tiempo: {epoch_time:.2f}s")
            print("─" * 50)

        # 8. Resultados finales
        return model, device, test_loader, test_dataset, expected_classes, all_probs, all_labels

    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Ejecutar entrenamiento
        model, device, test_loader, test_dataset, expected_classes, all_probs, all_labels = run_cnn_training()

        print("\n🎉 Entrenamiento completado!")
        model.eval()

        # Obtener IDs de imágenes de prueba - VERSIÓN CORREGIDA
        test_img_ids_used = []
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(test_loader):
                batch_ids = []
                for i in range(len(inputs)):
                    # Calcula el índice global en el dataset
                    global_idx = batch_idx * BATCH_SIZE + i
                    if global_idx < len(test_dataset):  # Verificación de límites
                        img_path = test_dataset.samples[global_idx][0]
                        img_id = img_path.split('/')[-1].split('.')[0]
                        batch_ids.append(img_id)
                test_img_ids_used.extend(batch_ids)

        # Verificar consistencia
        if len(test_img_ids_used) != len(all_labels):
            print(f"⚠️ Advertencia: Número de predicciones ({len(all_labels)}) no coincide con imágenes usadas ({len(test_img_ids_used)})")
            min_len = min(len(test_img_ids_used), len(all_labels))
            test_img_ids_used = test_img_ids_used[:min_len]
            all_probs = all_probs[:min_len]
            all_labels = all_labels[:min_len]

        # Guardar predicciones
        np.save(CNN_PREDICTIONS_PATH, {
            'predictions': np.array(all_probs),
            'true_labels': np.array(all_labels),
            'test_img_ids': np.array(test_img_ids_used),
            'class_names': expected_classes
        })

        print("\n📊 Resultados Finales:")
        print(f"- Modelo guardado en: best_cnn_model.pth")
        print(f"- Predicciones guardadas en: {CNN_PREDICTIONS_PATH}")
        print(f"- Muestras de prueba procesadas: {len(all_labels)}")

    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        print("Posibles causas:")
        print("- Problemas con los archivos de imagen")
        print("- Metadata inconsistente")
        print("- Problemas de memoria (reduce BATCH_SIZE si es necesario)")