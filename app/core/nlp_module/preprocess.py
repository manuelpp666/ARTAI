# ================================================================
# preprocess.py — versión BPE optimizada para español
# ================================================================
import os
import json
import random
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# -----------------------------
# ENTRENAR O CARGAR TOKENIZER BPE
# -----------------------------
def construir_vocab(texto, ruta_vocab="bpe_tokenizer.json", vocab_size=15000):
    """
    Entrena un tokenizador BPE desde texto plano y devuelve tokenizer, stoi y itos.
    """
    # Normalizar texto
    texto = texto.replace("\u2026", "...")  # “…” -> ...
    texto = texto.replace("\r\n", "\n")     # saltos de línea uniformes
    texto = " " + texto                      # ✅ espacio inicial para ByteLevel
    # texto = texto.lower()  # opcional: descomentar si quieres todo en minúscula

    if os.path.exists(ruta_vocab):
        tokenizer = Tokenizer.from_file(ruta_vocab)
        print(f"📚 Tokenizer BPE cargado desde {ruta_vocab}")
    else:
        print("🚀 Entrenando nuevo tokenizer BPE...")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # Pre-tokenizer robusto para español
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "SECCION"],
            show_progress=True
        )

        tokenizer.train_from_iterator([texto], trainer)
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
# CREAR BATCHES
# -----------------------------
def crear_batches(datos_codificados, seq_len, batch_size, token_seccion_id, device='cpu', porcentaje_seccion=0.5):
    """
    Genera batches mezclando secuencias que empiezan en SECCION y secuencias continuas.
    - porcentaje_seccion: % de batches que empiezan en SECCION
    """
    entradas, objetivos = [], []

    # 1️⃣ Secuencias que empiezan en SECCION
    indices_seccion = [i for i, t in enumerate(datos_codificados) if t == token_seccion_id]
    for idx in indices_seccion:
        if idx + seq_len + 1 <= len(datos_codificados):
            inp = datos_codificados[idx:idx+seq_len]
            targ = datos_codificados[idx+1:idx+seq_len+1]
            entradas.append(inp)
            objetivos.append(targ)

    # 2️⃣ Secuencias continuas aleatorias
    num_seq_libres = int(len(entradas) * (1 - porcentaje_seccion) / max(porcentaje_seccion, 0.01))
    max_start = len(datos_codificados) - seq_len - 1
    for _ in range(num_seq_libres):
        start = random.randint(0, max_start)
        inp = datos_codificados[start:start+seq_len]
        targ = datos_codificados[start+1:start+seq_len+1]
        entradas.append(inp)
        objetivos.append(targ)

    # Mezclar
    combinado = list(zip(entradas, objetivos))
    random.shuffle(combinado)

    # Crear batches
    for i in range(0, len(combinado), batch_size):
        batch = combinado[i:i+batch_size]
        x_batch = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y_batch = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        yield x_batch, y_batch