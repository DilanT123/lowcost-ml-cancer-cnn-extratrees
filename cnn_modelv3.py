import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
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
BATCH_SIZE = 32
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
IMAGE_SIZE = 224
NUM_EPOCHS = 30
PATIENCE = 3

# --- Transformaciones mejoradas ---
def get_transforms(augment=True):
    base_transforms = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if augment:
        augment_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ]
        return transforms.Compose(augment_transforms + base_transforms)
    return transforms.Compose(base_transforms)

# --- Dataset mejorado ---
class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = self._load_valid_samples()
        self._report_stats()
    
    def _load_valid_samples(self):
        valid_samples = []
        for _, row in self.df.iterrows():
            img_path = self._find_image_path(row['img_id_base'])
            if img_path:
                valid_samples.append((img_path, row['diagnostic_numeric']))
        return valid_samples
    
    def _find_image_path(self, img_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(self.img_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    return img_path
                except:
                    continue
        return None
    
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
        except:
            # Imagen dummy normalizada
            dummy_img = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE)
            dummy_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(dummy_img)
            return dummy_img, label
    
    def _report_stats(self):
        print(f"\n📊 Estadísticas del Dataset:")
        print(f"- Registros totales: {len(self.df)}")
        print(f"- Muestras válidas: {len(self.samples)}")
        
        if len(self.samples) > 0:
            class_counts = pd.Series([label for _, label in self.samples]).value_counts().sort_index()
            print("\n📈 Distribución de clases:")
            print(class_counts.to_string())

# --- Modelo ResNet18 Optimizado ---
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SkinLesionModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Congelar capas inicialmente
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reemplazar capa fully connected
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# --- Función Principal ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    print(f"\n⚙️ Configuración:")
    print(f"- Dispositivo: {device}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Épocas: {NUM_EPOCHS}")

    # 1. Cargar y preparar datos
    metadata_df = pd.read_csv('processed_metadata.csv')
    metadata_df['img_id_base'] = metadata_df['img_id'].str.replace(r'\.(png|jpg|jpeg)$', '', regex=True)
    metadata_df['diagnostic_numeric'] = pd.to_numeric(metadata_df['diagnostic_numeric'])
    metadata_df = metadata_df.dropna(subset=['diagnostic_numeric'])
    
    train_idx = np.load('train_indices.npy')
    test_idx = np.load('test_indices.npy')
    train_df = metadata_df.iloc[train_idx].reset_index(drop=True)
    test_df = metadata_df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Crear datasets y dataloaders
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, transform=get_transforms(augment=True))
    test_dataset = SkinLesionDataset(test_df, IMAGE_DIR, transform=get_transforms(augment=False))
    
    # 3. Balanceo de clases
    y_train = np.array([label for _, label in train_dataset.samples])
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    sampler = WeightedRandomSampler(weights=class_weights[y_train], num_samples=len(y_train), replacement=True)
    
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
        pin_memory=True
    )

    # 4. Modelo y optimización
    model = SkinLesionModel(num_classes=len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # 5. Entrenamiento
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping en época {epoch+1}")
                break

    return all_probs, all_labels

if __name__ == "__main__":
    all_probs, all_labels = train_model()
    
    # Guardar predicciones
    np.save(CNN_PREDICTIONS_PATH, {
        'predictions': np.array(all_probs),
        'true_labels': np.array(all_labels)
    })
    
    print("\n✅ Entrenamiento completado")
    print(f"- Modelo guardado: best_cnn_model.pth")
    print(f"- Predicciones guardadas: {CNN_PREDICTIONS_PATH}")