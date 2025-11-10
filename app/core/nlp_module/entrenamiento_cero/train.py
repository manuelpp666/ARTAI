# ================================================================
# entrenamiento_cero: Esto se ejecutó en la plataforma de kaggle
# ================================================================
import re
import json
from pathlib import Path


# -------------------------------------------------
# Transforma el dataset .txt a json
# -------------------------------------------------
def parse_wiki_txt(file_path):
    """
    Parsea el archivo .txt con formato:
    SECCION Título SECCION Contenido [FIN_SECCION]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Patrón corregido: captura título y contenido entre SECCION ... SECCION ... [FIN_SECCION]
    pattern = r'SECCION\s+(.*?)\s+SECCION\s+(.*?)\s*\[FIN_SECCION\]'
    sections = re.findall(pattern, text, re.DOTALL)
    
    data = []
    for title, content in sections:
        # Limpieza del contenido
        clean_content = content.strip()
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normaliza espacios
        clean_title = title.strip()
        
        data.append({
            'title': clean_title,
            'content': clean_content
        })
    
    print(f"Parseadas {len(data)} secciones correctamente.")
    return data

# === EJECUCIÓN EN KAGGLE ===
file_path = '/kaggle/input/dataset-arte/dataset_completo.txt'

# Verifica que el archivo existe
if not Path(file_path).exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

# Parsea el dataset
data = parse_wiki_txt(file_path)

# Muestra las primeras 2 entradas como ejemplo
for i, entry in enumerate(data[:2]):
    print(f"\n--- Entrada {i+1} ---")
    print(f"Título: {entry['title']}")
    print(f"Contenido (primeros 200 chars): {entry['content'][:200]}...")

# Opcional: Guarda como JSON para uso posterior (más fácil de cargar)
output_path = '/kaggle/working/arte_dataset_parsed.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nDataset parseado y guardado en: {output_path}")


# ================================================================
# Carga el archivo JSON parseado para entrenamiento
# ================================================================

# === 1. CARGAR EL JSON PARSEADO ===
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random

# Cargar el JSON que ya generaste
with open('/kaggle/working/arte_dataset_parsed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Dataset cargado: {len(data)} artículos de arte")

# ================================================================
# Construye y guarda el vocabulario
# ================================================================
import json
import pickle
from collections import Counter
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch

# -------------------------------
# 1. CARGAR EL JSON PARSEADO
# -------------------------------
JSON_PATH = "/kaggle/working/arte_dataset_parsed.json"

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"No se encontró {JSON_PATH}. ¿Ejecutaste el parser primero?")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"JSON cargado: {len(data)} documentos")

# -------------------------------
# 2. CONSTRUIR VOCABULARIO
# -------------------------------
print("Construyendo vocabulario...")

all_text = " ".join([item['content'] for item in data]).lower()
words = all_text.split()
word_counts = Counter(words)

# Tokens especiales + top 15,000
vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [
    word for word, count in word_counts.most_common(15000)
]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
seq_length = 60  # debe coincidir con tu modelo

print(f"Vocabulario creado: {vocab_size} palabras (seq_length={seq_length})")

# -------------------------------
# 3. GUARDAR vocab.pkl
# -------------------------------
VOCAB_DIR = "/kaggle/working/vocab"
os.makedirs(VOCAB_DIR, exist_ok=True)

vocab_path = os.path.join(VOCAB_DIR, "vocab.pkl")
with open(vocab_path, "wb") as f:
    pickle.dump({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'seq_length': seq_length
    }, f)

print(f"vocab.pkl guardado en: {vocab_path}")

# -------------------------------
# 4. (Opcional) CREAR arte_test.pkl
# -------------------------------
# === 4. CREAR arte_test.pkl===
class ArtTextDataset(Dataset):
    def __init__(self, data, seq_length, word2idx):
        self.sequences = []
        for item in data:
            tokens = [word2idx.get(w, word2idx['<UNK>']) for w in item['content'].lower().split()]
            if len(tokens) > seq_length + 1:
                for i in range(0, len(tokens) - seq_length, seq_length // 2):
                    seq = tokens[i:i + seq_length]
                    target = tokens[i + 1:i + seq_length + 1]
                    self.sequences.append((seq, target))  # ← listas, no tensores
    
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): 
        seq, tgt = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(tgt)  # ← aquí SÍ convertimos a tensor

# ... (split, etc.)

test_dataset = ArtTextDataset(test_data, seq_length, word2idx)

# GUARDAR COMO LISTAS (no necesitas .tolist())
test_data_save = {
    'input': [seq for seq, _ in test_dataset.sequences],
    'target': [tgt for _, tgt in test_dataset.sequences]
}

TEST_DIR = "/kaggle/working/data/processed"
os.makedirs(TEST_DIR, exist_ok=True)
test_path = os.path.join(TEST_DIR, "arte_test.pkl")
with open(test_path, "wb") as f:
    pickle.dump(test_data_save, f)

print(f"arte_test.pkl creado: {len(test_dataset)} secuencias → {test_path}")

# -------------------------------
# 5. VERIFICAR QUE TODO ESTÉ LISTO
# -------------------------------
print("\nArchivos generados en /kaggle/working/:")
!ls -R /kaggle/working/vocab /kaggle/working/data



# -------------------------------
# . Construir el dataset para LSTM
# -------------------------------
class ArtTextDataset(Dataset):
    def __init__(self, data, seq_length=60):
        self.seq_length = seq_length
        self.sequences = []
        
        for item in data:
            text = item['content'].lower()
            tokens = [word2idx.get(word, word2idx['<UNK>']) for word in text.split()]
            if len(tokens) > seq_length + 1:
                for i in range(0, len(tokens) - seq_length, seq_length // 2):
                    seq = tokens[i:i + seq_length]
                    target = tokens[i + 1:i + seq_length + 1]
                    self.sequences.append((seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)

# Crear dataset
seq_length = 60
dataset = ArtTextDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

print(f"Secuencias de entrenamiento: {len(dataset)}")


# === 4. MODELO LSTM ===
class LSTMFromScratch(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Instanciar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMFromScratch(vocab_size).to(device)

print(f"Modelo LSTM creado: {sum(p.numel() for p in model.parameters()):,} parámetros")
print(f"Dispositivo: {device}")


# === ENTRENAMIENTO PROLONGADO CON EARLY STOPPING ===
import os
from copy import deepcopy

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # LR baja cada 5 épocas

# Early stopping
best_loss = float('inf')
patience = 5
patience_counter = 0
max_epochs = 30
best_model_state = None

print("Iniciando entrenamiento prolongado (hasta 30 épocas)...\n")

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    epoch_steps = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        epoch_steps += 1
        
        if i % 400 == 0:
            print(f"  Época {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    # Promedio de loss
    avg_loss = total_loss / epoch_steps
    scheduler.step()
    
    print(f"\nÉPOCA {epoch+1}/{max_epochs} COMPLETADA")
    print(f"  Loss promedio: {avg_loss:.4f}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    
    
    # === EARLY STOPPING ===
    if avg_loss < best_loss - 0.001:  # Mejora significativa
        best_loss = avg_loss
        best_model_state = deepcopy(model.state_dict())
        patience_counter = 0
        print("  MEJOR MODELO GUARDADO!")
    else:
        patience_counter += 1
        print(f"  Sin mejora ({patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"\nEARLY STOPPING en época {epoch+1}")
        break

# === GUARDAR MEJOR MODELO ===
if best_model_state is not None:
    model.load_state_dict(best_model_state)

final_path = "/kaggle/working/lstm_arte_mejor.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'word2idx': word2idx,
    'idx2word': idx2word,
    'final_epoch': epoch + 1,
    'best_loss': best_loss
}, final_path)

print(f"\nMEJOR MODELO guardado en: {final_path}")
print(f"Loss final: {best_loss:.4f} en época {epoch+1}")

