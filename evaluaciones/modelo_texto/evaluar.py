# evaluaciones/modelo_texto/evaluar.py

import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ================================
# 1. CARGAR CHECKPOINT
# ================================
model_path = "../../models/arte/entrenamiento_desde_cero/v_cero.pth"
checkpoint = torch.load(model_path, map_location='cpu')
print("Claves en checkpoint:", list(checkpoint.keys()))

# -------------------------------
# EXTRAER VOCABULARIO
# -------------------------------
if 'word2idx' in checkpoint:
    word2idx = checkpoint['word2idx']
elif 'vocab' in checkpoint and isinstance(checkpoint['vocab'], list):
    vocab_list = checkpoint['vocab']
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
else:
    raise KeyError("No se encontró 'word2idx' ni lista 'vocab' válida")

vocab_size = len(word2idx)
pad_idx = word2idx.get('<PAD>', 0)
print(f"Vocabulario: {vocab_size} tokens | PAD index: {pad_idx}")

# ================================
# 2. MODELO
# ================================
class LSTMFromScratch(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
   
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMFromScratch(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Modelo cargado: {sum(p.numel() for p in model.parameters()):,} parámetros")


# ================================
# 3. CARGAR Y APLANAR DATOS DE PRUEBA
# ================================
test_data_path = "../../models/arte/entrenamiento_desde_cero/arte_test.pkl"
with open(test_data_path, 'rb') as f:
    test_data_raw = pickle.load(f)

# Aplanar si es lista de listas o diccionario de listas
all_sequences = []

if isinstance(test_data_raw, dict):
    for key, value in test_data_raw.items():
        if isinstance(value, list):
            all_sequences.extend(value)
        else:
            all_sequences.append(value)
elif isinstance(test_data_raw, list):
    for item in test_data_raw:
        if isinstance(item, list):
            all_sequences.extend(item)
        else:
            all_sequences.append(item)

test_data = all_sequences
print(f"Total de secuencias aplanadas: {len(test_data)}")

# Depuración
# Depuración ligera
print(f"Primer elemento: Lista de {len(test_data[0])} secuencias tokenizadas")
print(f"Tipo del primer elemento: {type(test_data[0])}")

# ================================
# 4. COLLATE FUNCTION ROBUSTA
# ================================
def collate_fn(batch):
    seqs = []
    for item in batch:
        if isinstance(item, dict):
            seq = None
            for key in ['tokens', 'input_ids', 'input', 'seq', 'token_ids']:
                if key in item:
                    seq = item[key]
                    break
            if seq is None:
                raise ValueError(f"No se encontró secuencia en: {item}")
        elif isinstance(item, (list, torch.Tensor)):
            seq = item
        else:
            raise TypeError(f"Tipo no soportado: {type(item)}")

        if isinstance(seq, list):
            seqs.append(torch.tensor(seq, dtype=torch.long))
        elif isinstance(seq, torch.Tensor):
            seqs.append(seq.to(torch.long))
        else:
            seqs.append(torch.tensor(seq, dtype=torch.long))

    inputs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    targets = inputs.clone()
    return inputs.to(device), targets.to(device)

# ================================
# 5. CREAR DATALOADER
# ================================
batch_size = 32
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ================================
# 6. FUNCIÓN DE EVALUACIÓN
# ================================
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print("\nIniciando evaluación...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluando"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs[:, :-1])
            target = targets[:, 1:].contiguous().view(-1)
            output = output.contiguous().view(-1, vocab_size)

            loss = criterion(output, target)

            num_tokens = target.size(0)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity, total_tokens

# ================================
# 7. EJECUTAR EVALUACIÓN
# ================================
print("\n" + "="*60)
print("EVALUACIÓN FINAL DEL MODELO (ENTRENADO DESDE CERO)")
print("="*60)

loss, perplexity, total_tokens = evaluate(model, test_loader)

print(f"\nRESULTADOS:")
print(f"  Loss (Cross-Entropy):     {loss:.4f}")
print(f"  Perplexity:               {perplexity:.2f}")
print(f"  Tokens evaluados:         {int(total_tokens):,}")
print(f"  Secuencias de prueba:     {len(test_data)}")
print(f"  Vocabulario:              {vocab_size:,} tokens")