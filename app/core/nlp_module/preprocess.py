import torch
import random
import os
import json

# -----------------------------
# FUNCIONES DE VOCABULARIO
# -----------------------------
def construir_vocab(texto):
    """Construye vocab dinámico a partir del texto."""
    chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def guardar_vocab(stoi, itos, ruta_modelo):
    """Guarda el vocabulario junto al modelo, en formato JSON."""
    ruta_vocab = ruta_modelo.replace(".pth", "_vocab.json")
    with open(ruta_vocab, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocabulario guardado en: {ruta_vocab}")


def cargar_vocab(ruta_modelo):
    """Carga vocabulario guardado en JSON."""
    ruta_vocab = ruta_modelo.replace(".pth", "_vocab.json")
    if not os.path.exists(ruta_vocab):
        raise FileNotFoundError(f"No se encontró vocabulario en {ruta_vocab}")
    with open(ruta_vocab, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convertir claves numéricas de itos a int
    itos = {int(k): v for k, v in data["itos"].items()}
    return data["stoi"], itos


# -----------------------------
# FUNCIONES DE CODIFICACIÓN
# -----------------------------
def codificar(texto, stoi):
    """Convierte texto a índices."""
    return [stoi.get(c, 0) for c in texto]


def decodificar(indices, itos):
    """Convierte índices a texto."""
    return "".join([itos.get(i, "?") for i in indices])


# -----------------------------
# FUNCIÓN PARA CREAR BATCHES
# -----------------------------
def crear_batches(datos, longitud_seq, tamaño_batch, device='cpu'):
    """Crea lotes tensoriales para el entrenamiento."""
    entradas, objetivos = [], []
    for frase in datos:
        for i in range(0, len(frase)-1, longitud_seq):
            inp = frase[i:i+longitud_seq]
            targ = frase[i+1:i+longitud_seq+1]
            if len(inp) < longitud_seq:
                inp += [0]*(longitud_seq-len(inp))
                targ += [0]*(longitud_seq-len(targ))
            entradas.append(inp)
            objetivos.append(targ)

    combinado = list(zip(entradas, objetivos))
    random.shuffle(combinado)
    for i in range(0, len(combinado), tamaño_batch):
        batch = combinado[i:i+tamaño_batch]
        x_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        yield x_batch, y_batch
