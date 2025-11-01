# ================================================================
# train_debug.py — modo diagnóstico para detectar NaN
# ================================================================
import os, torch, torch.nn as nn, torch.optim as optim
from torch import amp
from preprocess import construir_vocab, codificar, crear_batches
from transformer import Transformer
from contextlib import nullcontext

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/arte_traducido/dataset_completo.txt")
ruta_modelo_drive = "/content/drive/MyDrive/arte_chatbot/models/transformer_art_model_debug.pth"

# -------------------- DATA --------------------
with open(ruta_dataset, "r", encoding="utf-8") as f:
    texto = f.read().lower()
chars, stoi, itos = construir_vocab(texto)
vocab_size = len(chars)
data = [codificar(texto, stoi)]

seq_len, batch_size = 64, 4
modelo = Transformer(vocab_size=vocab_size).to(device)
criterio = nn.CrossEntropyLoss()
opt = optim.AdamW(modelo.parameters(), lr=5e-5, weight_decay=0.01)

use_amp = False  # ❌ Desactiva AMP mientras depuras
scaler = amp.GradScaler() if (device.type == "cuda" and use_amp) else None
ctx = amp.autocast("cuda") if (device.type == "cuda" and use_amp) else nullcontext()

# -------------------- DEBUG LOOP --------------------
for i_batch, (x_batch, y_batch) in enumerate(crear_batches(data, seq_len, batch_size, device)):
    with ctx:
        print("Input stats:", x_batch.min().item(), x_batch.max().item(), x_batch.dtype, x_batch.shape)
        print("Vocab size:", vocab_size)
        if torch.isnan(x_batch).any():
            raise RuntimeError("NaN en entrada del modelo")

        salida = modelo(x_batch)

        if torch.isnan(salida).any() or torch.isinf(salida).any():
            print("❌ NaN/Inf en logits. Min:", salida.min().item(), "Max:", salida.max().item())
            raise RuntimeError("NaN en salida del modelo")

        perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1))
        if torch.isnan(perdida) or torch.isinf(perdida):
            print("❌ NaN/Inf en pérdida:", perdida.item())
            raise RuntimeError("NaN en pérdida")

        print(f"Lote {i_batch} OK | Loss={perdida.item():.5f}")

        perdida.backward()

        # revisar gradientes
        for n, p in modelo.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f"❌ NaN/Inf en gradiente de {n}")
                raise RuntimeError("NaN en gradiente")

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step(); opt.zero_grad()

    if i_batch > 10:
        print("✅ Prueba de 10 batches sin NaN completada.")
        break