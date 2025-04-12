import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

AUDIO_LENGTH = 3 
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DysarthriaDataset(Dataset):
    def __init__(self, df, augment=False):
        self.valid_indices = []
        for idx in range(len(df)):
            file_path = df.iloc[idx]['file_path']
            if os.path.exists(file_path):
                try:
                    torchaudio.load(file_path)
                    self.valid_indices.append(idx)
                except:
                    print(f"Skipping corrupted file: {file_path}")
        self.df = df.iloc[self.valid_indices]
        self.augment = augment
        self.mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.time_mask = TimeMasking(time_mask_param=15)  
        self.freq_mask = FrequencyMasking(freq_mask_param=15)  
        
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        target_length = SAMPLE_RATE * AUDIO_LENGTH
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
            
        return waveform
    
    def _add_noise(self, waveform):
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise
    
    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]['file_path']
        label = self.df.iloc[idx]['label']
        waveform = self._load_audio(file_path)
        
        mel = self.mel_transform(waveform)
        mel = torchaudio.functional.amplitude_to_DB(
            mel, multiplier=10, amin=1e-10, db_multiplier=0
        )
        
        if self.augment:
            if torch.rand(1) > 0.5:
                mel = self.time_mask(mel)
            if torch.rand(1) > 0.5:
                mel = self.freq_mask(mel)
            if torch.rand(1) > 0.5:
                mel = self._add_noise(mel)
        
        return mel, torch.tensor(label, dtype=torch.float32)

class DysarthriaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.lstm1 = nn.LSTM(64 * (N_MELS//4), 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        batch_size, _, n_mels, n_time = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, n_time, -1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :] 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

def create_datasets(data_dir):
    classes = ['non_dysarthria', 'dysarthria']
    genders = ['male', 'female']
    file_paths = []
    labels = []
    demographics = []

    for class_name in classes:
        for gender in genders:
            class_dir = os.path.join(data_dir, class_name, gender)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith('.wav'):
                        file_paths.append(os.path.join(root, file))
                        labels.append(1 if class_name == 'dysarthria' else 0)
                        demographics.append(gender)

    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'gender': demographics
    })
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[['label', 'gender']],
        random_state=42
    )
    
    return train_df, test_df

def train_model(model, train_loader, test_loader):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    best_auc = 0
    history = {'train_loss': [], 'test_loss': [], 'auc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        auc = roc_auc_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['auc'].append(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | AUC: {auc:.4f}')
    
    return history

def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds.extend(torch.round(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    data_dir = "hills_prj/voice_data"
    train_df, test_df = create_datasets(data_dir)
    train_dataset = DysarthriaDataset(train_df, augment=True)
    test_dataset = DysarthriaDataset(test_df, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    model = DysarthriaDetector()

    history = train_model(model, train_loader, test_loader)
    
    evaluate_model(model, test_loader)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['auc'], label='AUC')
    plt.title('Validation AUC')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()