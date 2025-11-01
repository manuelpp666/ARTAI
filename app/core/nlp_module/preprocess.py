# ================================================================
# preprocess.py — versión con tokenización BPE
# ================================================================
import os
import json
import random
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# -----------------------------
# ENTRENAR TOKENIZER BPE
# -----------------------------
def construir_vocab(texto, ruta_vocab="bpe_tokenizer.json", vocab_size=8000):
    """
    Entrena un tokenizador BPE desde texto plano y devuelve stoi/itos compatibles.
    """
    if os.path.exists(ruta_vocab):
        tokenizer = Tokenizer.from_file(ruta_vocab)
        print(f"📚 Tokenizer BPE cargado desde {ruta_vocab}")
    else:
        print("🚀 Entrenando nuevo tokenizer BPE...")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator([texto], trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.save(ruta_vocab)
        print(f"✅ Tokenizer BPE entrenado y guardado en {ruta_vocab}")

    vocab = tokenizer.get_vocab()
    stoi = vocab
    itos = {i: s for s, i in vocab.items()}
    return tokenizer, stoi, itos

# -----------------------------
# GUARDAR / CARGAR VOCABULARIO
# -----------------------------
def guardar_vocab(stoi, itos, ruta_modelo):
    ruta_vocab = ruta_modelo.replace(".pth", "_vocab.json")
    with open(ruta_vocab, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocabulario guardado en: {ruta_vocab}")

def cargar_vocab(ruta_modelo):
    ruta_vocab = ruta_modelo.replace(".pth", "_vocab.json")
    if not os.path.exists(ruta_vocab):
        raise FileNotFoundError(f"No se encontró vocabulario en {ruta_vocab}")
    with open(ruta_vocab, "r", encoding="utf-8") as f:
        data = json.load(f)
    itos = {int(k): v for k, v in data["itos"].items()}
    return data["stoi"], itos

# -----------------------------
# CODIFICACIÓN / DECODIFICACIÓN
# -----------------------------
def codificar(texto, tokenizer):
    return tokenizer.encode(texto).ids

def decodificar(indices, tokenizer):
    return tokenizer.decode(indices)

# -----------------------------
# CREAR BATCHES
# -----------------------------
def crear_batches(datos, longitud_seq, tamaño_batch, device='cpu'):
    entradas, objetivos = [], []
    for i in range(0, len(datos) - longitud_seq - 1, longitud_seq):
        inp = datos[i:i+longitud_seq]
        targ = datos[i+1:i+longitud_seq+1]
        entradas.append(inp)
        objetivos.append(targ)

    combinado = list(zip(entradas, objetivos))
    random.shuffle(combinado)

    for i in range(0, len(combinado), tamaño_batch):
        batch = combinado[i:i+tamaño_batch]
        x_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        yield x_batch, y_batch
