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
        try:
            tokenizer = Tokenizer.from_file(ruta_vocab)
            print(f"📚 Tokenizer BPE cargado desde {ruta_vocab}")
        except Exception as e:
            print(f"Error al cargar tokenizer (probablemente corrupto): {e}")
            print("Se entrenará uno nuevo...")
            os.remove(ruta_vocab) # Borra el archivo corrupto
            return construir_vocab(ruta_dataset, ruta_vocab, vocab_size) # Vuelve a intentarlo
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
            # ✅ CORRECCIÓN 1: Usamos 'utf-8-sig'.
            # Esto lee UTF-8 estándar y también maneja el 'BOM' (un caracter
            # invisible que Windows a veces añade) que puede corromper 'utf-8'.
            # También añadimos 'errors="ignore"' como red de seguridad final.
            with open(ruta_dataset, "r", encoding="utf-8-sig", errors="ignore") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    # Reemplazamos caracteres problemáticos comunes
                    yield chunk.replace("\u2026", "...").replace("\r\n", "\n")

        tokenizer.train_from_iterator(iter_texto(ruta_dataset), trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # Guardamos sin 'pretty=True' para evitar corrupción
        tokenizer.save(ruta_vocab, pretty=False) 
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
    # ✅ CORRECCIÓN 2: Los archivos JSON SIEMPRE deben guardarse como 'utf-8'.
    with open(ruta_vocab, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocabulario guardado en: {ruta_vocab}")

def cargar_vocab(ruta_modelo):
    ruta_vocab = ruta_modelo.replace(".pth", "_vocab.json")
    if not os.path.exists(ruta_vocab):
        raise FileNotFoundError(f"No se encontró vocabulario en {ruta_vocab}")
    # ✅ CORRECCIÓN 3: Los archivos JSON SIEMPRE deben leerse como 'utf-8'.
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
    lineas = []
    # Obtener líneas según tipo
    if isinstance(input_data, str):
        # ✅ CORRECCIÓN 4: Usamos 'utf-8-sig' aquí también, para leer
        # los archivos de entrenamiento/validación.
        with open(input_data, "r", encoding="utf-8-sig", errors="ignore") as f:
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