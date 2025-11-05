# ================================================================
# preprocess.py — versión BPE optimizada para español (streaming)
# ================================================================
import os
import json
import random
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# -----------------------------
# ENTRENAR O CARGAR TOKENIZER BPE
# -----------------------------
def construir_vocab(ruta_dataset, ruta_vocab="bpe_tokenizer.json", vocab_size=15000, chunk_size=1024*1024):
    """
    Entrena un tokenizador BPE desde un archivo de texto por streaming.
    Devuelve tokenizer, stoi y itos.
    """
    if os.path.exists(ruta_vocab):
        tokenizer = Tokenizer.from_file(ruta_vocab)
        print(f"📚 Tokenizer BPE cargado desde {ruta_vocab}")
    else:
        print("🚀 Entrenando nuevo tokenizer BPE por streaming...")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # Pre-tokenizer robusto para español
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "SECCION", "[FIN_SECCION]"],
            show_progress=True
        )

        # Generador que devuelve trozos del dataset
        def iter_texto(ruta_dataset, chunk_size=chunk_size):
            # ✅ CORRECCIÓN: Usamos 'latin-1' en lugar de 'utf-8'.
            # Esta codificación es más robusta para textos en español 
            # y rara vez falla, preservando 'ñ' y tildes.
            with open(ruta_dataset, "r", encoding="latin-1") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    # Reemplazamos caracteres problemáticos comunes
                    yield chunk.replace("\u2026", "...").replace("\r\n", "\n")

        tokenizer.train_from_iterator(iter_texto(ruta_dataset), trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.save(ruta_vocab)
        print(f"✅ Tokenizer BPE entrenado y guardado en {ruta_vocab}")

    # Crear mappings stoi / itos
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
    texto = " " + texto  # ✅ agregar espacio inicial para codificación
    return tokenizer.encode(texto).ids

def decodificar(indices, tokenizer):
    texto = tokenizer.decode(indices)
    return texto.lstrip()  # eliminar espacio inicial sobrante

# -----------------------------
# GENERAR BATCHES POR STREAMING
# -----------------------------
def generar_batches(input_data, tokenizer, seq_len, batch_size, token_seccion_id, device='cpu'):
    """
    Genera batches de forma dinámica por streaming.
    input_data puede ser un path (str) o una lista de líneas
    """
    # Obtener líneas según tipo
    if isinstance(input_data, str):
        with open(input_data, "r", encoding="utf-8") as f:
            lineas = f.readlines()
    elif isinstance(input_data, list):
        lineas = input_data
    else:
        raise TypeError("input_data debe ser str o list")
    
    buffer = []
    batch_x, batch_y = [], []

    # Iterar sobre las líneas ya leídas
    for linea in lineas:
        if not linea.strip().endswith("[FIN_SECCION]"):
            linea = linea.strip() + " [FIN_SECCION]"
        buffer.extend(tokenizer.encode(" " + linea).ids)
        
        while len(buffer) >= seq_len + 1:
            x = buffer[:seq_len]
            y = buffer[1:seq_len+1]
            buffer = buffer[seq_len:]

            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) == batch_size:
                x_tensor = torch.tensor(batch_x, dtype=torch.long, device=device)
                y_tensor = torch.tensor(batch_y, dtype=torch.long, device=device)
                yield x_tensor, y_tensor
                batch_x, batch_y = [], []
