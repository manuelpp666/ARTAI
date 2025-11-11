import torch
import torch.nn as nn
import pickle
import os

# ==============================================
# 1. DEFINICIÓN DEL MODELO CORRECTO (LSTM)
# ==============================================
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


# ==============================================
# 2. FUNCIÓN PARA CARGAR EL MODELO GUARDADO
# ==============================================
def cargar_modelo_desde_cero(model_path):
    """Carga el checkpoint guardado en entrenamiento_cero (LSTM)"""
    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise ValueError("❌ El checkpoint no tiene 'model_state_dict' (¿seguro que es el archivo correcto?).")

    vocab = checkpoint["vocab"]
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]
    state_dict = checkpoint["model_state_dict"]

    vocab_size = len(vocab)
    model = LSTMFromScratch(vocab_size)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Modelo LSTM cargado correctamente ({vocab_size} palabras).")
    return model, word2idx, idx2word


# ==============================================
# 3. GENERADOR DE TEXTO (sampling)
# ==============================================
@torch.no_grad()
def generar_texto(model, word2idx, idx2word, prompt, max_len=60, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device

    words = prompt.lower().split()
    seq_length = 60
    input_ids = [word2idx.get(w, word2idx.get("<UNK>", 0)) for w in words[-seq_length:]]
    input_tensor = torch.tensor([input_ids]).to(device)

    generated = input_ids.copy()
    unk_idx = word2idx.get("<UNK>")
    eos_idx = word2idx.get("<EOS>", -1)

    for _ in range(max_len):
        output = model(input_tensor)
        next_word_logits = output[0, -1, :] / temperature
        probs = torch.softmax(next_word_logits, dim=-1)

        # Evitar generar <UNK>
        if unk_idx is not None:
            probs[unk_idx] = 0
            probs = probs / probs.sum()

        next_word_id = torch.multinomial(probs, 1).item()
        if next_word_id >= len(idx2word):
            next_word_id = 0
        if next_word_id == eos_idx:
            break

        generated.append(next_word_id)
        input_tensor = torch.tensor([generated[-seq_length:]]).to(device)

    palabras_salida = [
        idx2word.get(idx, "")
        for idx in generated[len(input_ids):]
        if idx2word.get(idx, "") not in ["<UNK>", "<EOS>", "", "<unk>"]
    ]
    respuesta = " ".join(palabras_salida).strip()

    return respuesta.capitalize() if respuesta else "No sé qué decir."
