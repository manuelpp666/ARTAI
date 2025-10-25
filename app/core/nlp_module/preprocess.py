# app/core/nlp_module/preprocess.py
import torch
import random

# Diccionario de ejemplo (en la práctica, generar dinámicamente a partir del corpus)
chars = sorted(list("abcdefghijklmnopqrstuvwxyzáéíóúü ,.!?\n"))
vocab_size = len(chars)
# Mappings de carácter a índice y viceversa
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# Función de codificación y decodificación
def codificar(texto):
    """Convierte texto a lista de índices."""
    return [stoi.get(c, 0) for c in texto]

def decodificar(indices):
    """Convierte lista de índices a texto."""
    return "".join([itos[i] for i in indices])

# Función para crear batches de entrenamiento
def crear_batches(datos, longitud_seq, tamaño_batch, device='cpu'):
    """Convierte lista de secuencias en batches tensoriales."""
    entradas, objetivos = [], []
    for frase in datos:
        for i in range(0, len(frase)-1, longitud_seq):
            inp = frase[i:i+longitud_seq]
            targ = frase[i+1:i+longitud_seq+1]
            # padding si la secuencia es más corta
            if len(inp) < longitud_seq:
                inp += [0]*(longitud_seq-len(inp))
                targ += [0]*(longitud_seq-len(targ))
            entradas.append(inp)
            objetivos.append(targ)
    # Mezclar los batches
    combinado = list(zip(entradas, objetivos))
    random.shuffle(combinado)
    for i in range(0, len(combinado), tamaño_batch):
        batch = combinado[i:i+tamaño_batch]
        x_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        yield x_batch, y_batch
