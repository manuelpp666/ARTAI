# ================================================================
# entrenamiento_cero: Esto se ejecutó en la plataforma de kaggle
# ================================================================
# Este archivo contiene:
# 1) Parser del dataset de texto plano -> JSON
# 2) Carga del JSON y conteo de vocabulario
# 3) Construcción de datasets (secuencias) para entrenamiento
# 4) Definición de un modelo LSTM desde cero (PyTorch)
# 5) Rutina de entrenamiento con Adam, scheduler y early stopping
# 6) Guardado del mejor modelo
#
# Comentarios incluidos para cada paso y cada bloque importante.
# ================================================================

import re                # expresiones regulares (para parsear el .txt)
import json              # para leer/escribir JSON
from pathlib import Path # utilidades de path (verificar existencia de archivos)


# -------------------------------------------------
# Transforma el dataset .txt a json
# -------------------------------------------------
def parse_wiki_txt(file_path):
    """
    Parsea el archivo .txt con formato:
    SECCION Título SECCION Contenido [FIN_SECCION]

    Explicación:
    - Usa expresiones regulares (regex) para extraer pares (título, contenido).
    - re.DOTALL permite que '.' coincida también con saltos de línea, necesario para capturar contenido multilínea.
    - Se normalizan espacios para limpiar saltos de línea/espacios múltiples.
    - Devuelve una lista de dicts: [{'title': ..., 'content': ...}, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Patrón corregido: captura título y contenido entre SECCION ... SECCION ... [FIN_SECCION]
    # Detalle del patrón:
    # - 'SECCION\\s+(.*?)\\s+SECCION\\s+(.*?)\\s*\\[FIN_SECCION\\]'
    # - (.*?)  → captura no codiciosa (mínima) para título y contenido
    # - \\s+   → al menos un espacio en blanco entre tokens
    # - \\[FIN_SECCION\\]  → marcador de fin
    pattern = r'SECCION\s+(.*?)\s+SECCION\s+(.*?)\s*\[FIN_SECCION\]'
    sections = re.findall(pattern, text, re.DOTALL)
    
    data = []
    for title, content in sections:
        # Limpieza del contenido: quitar espacios iniciales/finales y colapsar espacios múltiples
        clean_content = content.strip()
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Normaliza espacios y saltos de línea
        clean_title = title.strip()
        
        data.append({
            'title': clean_title,
            'content': clean_content
        })
    
    # Mensaje informativo (útil en notebook/kaggle para saber cuántas secciones parseó)
    print(f"Parseadas {len(data)} secciones correctamente.")
    return data

# === EJECUCIÓN EN KAGGLE ===
file_path = '/kaggle/input/dataset-arte/dataset_completo.txt'

# Verifica que el archivo existe (previene errores silenciosos)
if not Path(file_path).exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")

# Parsea el dataset: convierte texto crudo a lista de dicts
data = parse_wiki_txt(file_path)

# Muestra las primeras 2 entradas como ejemplo (útil para inspección rápida)
for i, entry in enumerate(data[:2]):
    print(f"\n--- Entrada {i+1} ---")
    print(f"Título: {entry['title']}")
    print(f"Contenido (primeros 200 chars): {entry['content'][:200]}...")

# Opcional: Guarda como JSON para uso posterior (más rápido y fácil de cargar que volver a parsear)
output_path = '/kaggle/working/arte_dataset_parsed.json'
with open(output_path, 'w', encoding='utf-8') as f:
    # ensure_ascii=False para mantener acentos y caracteres UTF-8 legibles
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nDataset parseado y guardado en: {output_path}")


# ================================================================
# Carga el archivo JSON parseado para entrenamiento
# ================================================================

# === 1. CARGAR EL JSON PARSEADO ===
# Importaciones relacionadas con PyTorch y manipulación de datos
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
import pickle
import os

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

# all_text: concatenación de todos los contenidos en minúsculas
all_text = " ".join([item['content'] for item in data]).lower()

# words: tokenización muy simple por espacios.
# Nota: Tokenizar por espacios es rápido y funciona, pero tiene limitaciones:
#   - No maneja puntuación como tokens separados ni subword tokenization.
#   - Para modelos más robustos se usan tokenizadores BPE/WordPiece/Subword.
words = all_text.split()

# word_counts: conteo de frecuencia por palabra (Counter)
word_counts = Counter(words)

# Tokens especiales: necesarios para padding, inicio/fin y desconocidos
# Seleccionamos top 15,000 palabras para limitar vocabulario (memoria y cómputo)
vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [
    word for word, count in word_counts.most_common(15000)
]
# Mapeos palabra->índice y índice->palabra
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# longitud fija de secuencia usada por el modelo (ventana de palabras)
seq_length = 60  # debe coincidir con tu modelo

print(f"Vocabulario creado: {vocab_size} palabras (seq_length={seq_length})")

# -------------------------------
# 3. GUARDAR vocab.pkl
# -------------------------------
VOCAB_DIR = "/kaggle/working/vocab"
os.makedirs(VOCAB_DIR, exist_ok=True)

vocab_path = os.path.join(VOCAB_DIR, "vocab.pkl")
with open(vocab_path, "wb") as f:
    # Guardamos word2idx e idx2word para poder usar el modelo luego (inferencia)
    pickle.dump({
        'word2idx': word2idx,
        'idx2word': idx2word,
        'seq_length': seq_length
    }, f)

print(f"vocab.pkl guardado en: {vocab_path}")

# -------------------------------
# 4. (Opcional) CREAR arte_test.pkl
# -------------------------------
# Esta sección crea un dataset de test/validación en formato listo para cargar.
# Crea ventanas solapadas (stride = seq_length // 2) para aprovechar más datos.
class ArtTextDataset(Dataset):
    def __init__(self, data, seq_length, word2idx):
        """
        Construye secuencias (input,target) para modelado de lenguaje:
        - input: tokens[i : i+seq_length]
        - target: tokens[i+1 : i+seq_length+1]
        Este objetivo enseña a predecir la próxima palabra en cada paso.
        """
        self.sequences = []
        for item in data:
            # tokenizamos por espacios y mapeamos a índices, usando <UNK> si no está la palabra
            tokens = [word2idx.get(w, word2idx['<UNK>']) for w in item['content'].lower().split()]
            # Necesitamos al menos seq_length + 1 tokens para crear input+target
            if len(tokens) > seq_length + 1:
                # Recorremos con stride para crear ejemplos solapados (aumenta datos efectivamente)
                for i in range(0, len(tokens) - seq_length, seq_length // 2):
                    seq = tokens[i:i + seq_length]
                    target = tokens[i + 1:i + seq_length + 1]
                    # guardamos pares como listas (se convertirán a tensores en __getitem__)
                    self.sequences.append((seq, target))
    
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): 
        seq, tgt = self.sequences[idx]
        # devolvemos tensores LongTensor (índices)
        return torch.tensor(seq), torch.tensor(tgt)

# ... (split, etc.)  # aquí en tu script original asumías test_data ya creada

# Ejemplo de cómo crear test_dataset (tu script necesita definir test_data previamente)
# test_dataset = ArtTextDataset(test_data, seq_length, word2idx)

# GUARDAR COMO LISTAS (no necesitas .tolist())
# test_data_save = {
#     'input': [seq for seq, _ in test_dataset.sequences],
#     'target': [tgt for _, tgt in test_dataset.sequences]
# }
# (Nota: guardar como pickle permite cargar rápidamente sin rehacer el preprocesamiento)

TEST_DIR = "/kaggle/working/data/processed"
os.makedirs(TEST_DIR, exist_ok=True)
# test_path = os.path.join(TEST_DIR, "arte_test.pkl")
# with open(test_path, "wb") as f:
#     pickle.dump(test_data_save, f)
# print(f"arte_test.pkl creado: {len(test_dataset)} secuencias → {test_path}")

# -------------------------------
# 5. VERIFICAR QUE TODO ESTÉ LISTO
# -------------------------------
print("\nArchivos generados en /kaggle/working/:")
# Nota: el uso de !ls es específico de notebooks; si ejecutas como script podrías usar os.listdir
try:
    # En entornos de notebook (Kaggle) esta línea listará archivos. En script normal se ignora o falla.
    get_ipython  # type: ignore
    !ls -R /kaggle/working/vocab /kaggle/working/data
except Exception:
    # En modo script no hacemos nada especial
    pass


# -------------------------------
# Construir el dataset para LSTM (de nuevo, con clase simplificada)
# -------------------------------
class ArtTextDataset(Dataset):
    def __init__(self, data, seq_length=60):
        """
        Versión del dataset usada para entrenamiento final.
        Igual que la anterior: ventanas solapadas, target desplazado una posición.
        """
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

# Crear dataset y DataLoader (baraja y agrupa por batch)
seq_length = 60
dataset = ArtTextDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

print(f"Secuencias de entrenamiento: {len(dataset)}")


# === 4. MODELO LSTM ===
class LSTMFromScratch(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        """
        Arquitectura:
        - Embedding: convierte índices en vectores densos (representaciones semánticas).
        - LSTM: red recurrente con memoria (capacidad de manejar dependencias largas).
        - FC: capa lineal que proyecta estados ocultos a logits sobre el vocabulario.
        - Dropout: regularización para evitar overfitting.
        """
        super().__init__()
        # Embedding: vocab_size -> embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # LSTM: procesa la secuencia de embeddings
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Capa final que produce un logit por palabra del vocabulario en cada paso de tiempo
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: LongTensor shape (batch_size, seq_length)
        Retorna: logits shape (batch_size, seq_length, vocab_size)
        """
        x = self.embedding(x)          # -> (batch_size, seq_length, embed_size)
        x = self.dropout(x)            # regularización
        out, _ = self.lstm(x)          # out -> (batch_size, seq_length, hidden_size)
        out = self.fc(out)             # -> (batch_size, seq_length, vocab_size)
        return out

# Instanciar modelo y mover a dispositivo disponible (GPU si hay)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMFromScratch(vocab_size).to(device)

# Contar parámetros te da una idea de la capacidad del modelo
print(f"Modelo LSTM creado: {sum(p.numel() for p in model.parameters()):,} parámetros")
print(f"Dispositivo: {device}")


# === ENTRENAMIENTO PROLONGADO CON EARLY STOPPING ===
import os
from copy import deepcopy

# ---- Configuración de entrenamiento ----
# CrossEntropyLoss es la pérdida estándar para clasificación multiclase:
# - Cada posición del tiempo es una clasificación entre vocab_size clases.
# - ignore_index=0 evita que el padding afecte la pérdida (si existiera).
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Adam: optimizador adaptativo, combina Momentum + RMSprop ideas.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler reduce el learning rate cada 'step_size' épocas multiplicándolo por 'gamma'.
# Esto ayuda a refinar la convergencia conforme avanza el entrenamiento.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # LR baja cada 5 épocas

# Early stopping: guarda el mejor modelo (por loss promedio) y detiene si no mejora
best_loss = float('inf')
patience = 5                # número de épocas sin mejora antes de parar
patience_counter = 0
max_epochs = 30
best_model_state = None

print("Iniciando entrenamiento prolongado (hasta 30 épocas)...\n")

# ---- Bucle de entrenamiento ----
for epoch in range(max_epochs):
    model.train()           # modo entrenamiento (activa dropout)
    total_loss = 0
    epoch_steps = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        # mover tensores a GPU/CPU según corresponda
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()                 # limpiar gradientes
        outputs = model(inputs)               # logits shape: (B, seq_length, vocab_size)
        
        # CrossEntropyLoss espera (N, C) vs (N), así que aplastamos las dimensiones:
        # outputs.view(-1, vocab_size) -> (B * seq_length, vocab_size)
        # targets.view(-1) -> (B * seq_length)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()                       # backpropagation
        # clip_grad_norm_: evita explosión de gradientes en RNNs (limita la norma del gradiente)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                      # actualización de parámetros
        
        total_loss += loss.item()
        epoch_steps += 1
        
        # Loggin periódico por batch (útil cuando hay muchos batches)
        if i % 400 == 0:
            print(f"  Época {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    # Promedio de loss por batch en la época
    avg_loss = total_loss / epoch_steps
    scheduler.step()  # actualizar learning rate según scheduler
    
    print(f"\nÉPOCA {epoch+1}/{max_epochs} COMPLETADA")
    print(f"  Loss promedio: {avg_loss:.4f}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # === EARLY STOPPING: criterio simple de mejora mínima ===
    # Si la mejora es mayor que 0.001 en avg_loss, consideramos que hay mejora significativa.
    if avg_loss < best_loss - 0.001:  # Mejora significativa (umbral)
        best_loss = avg_loss
        best_model_state = deepcopy(model.state_dict())  # guardamos pesos del mejor modelo
        patience_counter = 0
        print("  MEJOR MODELO GUARDADO!")
    else:
        patience_counter += 1
        print(f"  Sin mejora ({patience_counter}/{patience})")
    
    # Si no hay mejora por 'patience' épocas, detenemos el entrenamiento
    if patience_counter >= patience:
        print(f"\nEARLY STOPPING en época {epoch+1}")
        break

# === GUARDAR MEJOR MODELO ===
# Si guardamos best_model_state durante el entrenamiento, lo cargamos antes de persistir.
if best_model_state is not None:
    model.load_state_dict(best_model_state)

final_path = "/kaggle/working/lstm_arte_mejor.pth"
torch.save({
    'model_state_dict': model.state_dict(),  # pesos entrenados
    'vocab': vocab,                          # vocabulario (necesario para inferencia)
    'word2idx': word2idx,
    'idx2word': idx2word,
    'final_epoch': epoch + 1,
    'best_loss': best_loss
}, final_path)

print(f"\nMEJOR MODELO guardado en: {final_path}")
print(f"Loss final: {best_loss:.4f} en época {epoch+1}")
