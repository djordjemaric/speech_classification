#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataset import train_set, val_set, test_set, SubsetSC


# In[2]:


import torch
import torchaudio.transforms
from typing import List, Tuple
from torch.types import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[3]:


class Preprocessor:
    def __init__(self):
        melkwargs = {
            'n_mels': 80,
            'n_fft': 400,
            'hop_length': 160,
            'mel_scale': 'htk'
        }
        self.mfcc = torchaudio.transforms.MFCC(melkwargs=melkwargs)
        self.target_length = 16000
    def preprocess(self, waveform : Tensor) -> Tensor:
        if waveform.shape[0] != self.target_length:
            waveform = self._pad_trim(waveform)
        return self.mfcc(waveform)
    def _pad_trim(self, wav : Tensor) -> Tensor:
        size = wav.shape[0]
        if size < self.target_length:
            return torch.nn.functional.pad(wav, (0, self.target_length - size))
        else:
            return wav[:self.target_length]


# In[4]:


classes = []
with open('classes.txt', 'r') as f:
    classes = f.read().splitlines()
label_to_class = {}
class_to_label = {}
for idx, cl in enumerate(classes):
    label_to_class[cl] = idx
    class_to_label[idx] = cl


# In[5]:


def create_loader(dataset : SubsetSC, batch_size : int) -> torch.utils.data.DataLoader:
    preprocessor = Preprocessor()
    def _collate(batch):
        waveforms = []
        labels = []
        for waveform, sample_rate, label, *_ in batch:
            mfcc = preprocessor.preprocess(waveform)
            waveforms.append(mfcc)
            labels.append(torch.tensor(label_to_class[label]))
            
        # Pad sequences to same length
        waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms, 
            batch_first=True
        )
        labels = torch.stack(labels)
        return waveforms, labels
    return torch.utils.data.DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=_collate)


# In[6]:


class SpeechCNN(nn.Module):
    def __init__(self, num_classes=35, dropout_rate=0.5):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# In[7]:


class History:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.best_epoch = 0
        
    def update(self, epoch : int, train_loss : float, train_acc : float,
                     val_loss : float, val_acc : float, lr : float) -> bool:
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            return True
        return False
    
    def print_epoch(self, epoch : int, epochs : int) -> None:
        print(f"\nEpoch: {epoch+1}/{epochs}")
        print(f"Train Loss: {self.history['train_loss'][-1]:.4f} | "
              f"Train Acc: {self.history['train_acc'][-1]:.2f}%")
        print(f"Validation Loss: {self.history['val_loss'][-1]:.4f} | "
              f"Validation Acc: {self.history['val_acc'][-1]:.2f}%")
        print(f"Learning Rate: {self.history['lr'][-1]:.6f}")
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('metrics.png')
        plt.show()


# In[8]:


class CNNModel:
    def __init__(self):
        self.model = SpeechCNN()
        self.best_model = SpeechCNN()
        self.best_model.load_state_dict(torch.load(f='best_model.pth'))
        self.best_model.eval()
        self.history = History()
        self.early_stopping_patience = 5  
        self.early_stopping_min_delta = 0.001  
    def predict(self, test_loader : torch.utils.data.DataLoader, model : SpeechCNN = None, device : str = 'cpu') -> List[int]:
        model = model or self.best_model
        preds = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                preds.append(predicted)
        return torch.cat(preds)
    def predict_single(self, wav : Tensor, model : SpeechCNN = None) -> int:
        model = model or self.best_model
        with torch.no_grad():
            # wav = self.preprocess(wav)
            outputs = model(wav)
            _, predicted = outputs.max(1)
        return predicted

    def fit(self, train_loader : torch.utils.data.DataLoader,
            val_loader : torch.utils.data.DataLoader,
            epochs : int = 30,
            device : str = 'cpu'):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.005, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            min_lr=0.0001
        )
        
        self.model = self.model.to(device)
        # Add early stopping variables
        no_improve_count = 0
        min_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                   
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < (min_val_loss - self.early_stopping_min_delta):
                min_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.early_stopping_patience:
                print(f'\nEarly stopping triggered after epoch {epoch + 1}')
                break

            best = self.history.update(epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])
            self.history.print_epoch(epoch, epochs)
            
            torch.save(self.model.state_dict(), f'./models/model_{epoch}_{val_acc:.2f}_.pth')
            if best:
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.best_model = self.model