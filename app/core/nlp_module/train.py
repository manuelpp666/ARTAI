# ================================================================
# train_colab_progresivo.py — versión mejorada y estable
# ================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import construir_vocab, guardar_vocab, codificar, crear_batches
from transformer import Transformer
from generator import generar_texto
from torch import amp
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter
import numpy as np

# ================================================================
# ⚙️ Configuración general
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

seq_len = 192
batch_size = 8
accum_steps = 2
checkpoint_every = 2
porc_validacion = 0.1

# ================================================================
# 📁 Rutas
# ================================================================
base_dir = os.path.dirname(__file__)
ruta_dataset = os.path.join(base_dir, "../../../datasets/español/arte_traducido/dataset_completo.txt")
ruta_vocab = os.path.join(base_dir, "../../../models/vocab_art.pt")
ruta_modelo_drive = "/content/drive/MyDrive/arte_chatbot/models/transformer_art_model.pth"
os.makedirs(os.path.dirname(ruta_modelo_drive), exist_ok=True)

# ================================================================
# 📖 Dataset
# ================================================================
print("Leyendo dataset en:", ruta_dataset)
with open(ruta_dataset, "r", encoding="utf-8") as f:
    texto = f.read().lower()
print(f"Dataset completo: {len(texto)} caracteres.")

punto_corte = int(len(texto) * (1 - porc_validacion))
pos_final = max(texto.rfind('.', 0, punto_corte), texto.rfind('\n', 0, punto_corte))
if pos_final == -1: pos_final = punto_corte

texto_train, texto_val = texto[:pos_final], texto[pos_final:]
print(f"Entrenamiento: {len(texto_train)} chars | Validación: {len(texto_val)} chars")

# ================================================================
# 🧠 Vocabulario
# ================================================================
tokenizer, stoi, itos = construir_vocab(texto, ruta_vocab="bpe_tokenizer.json", vocab_size=8000)
guardar_vocab(stoi, itos, ruta_vocab)
vocab_size = len(stoi)

data_train = codificar(texto_train, tokenizer)
data_val = codificar(texto_val, tokenizer)

# ================================================================
# 🧩 Modelo y optimizador
# ================================================================
modelo = Transformer(vocab_size=vocab_size).to(device)
criterio = nn.CrossEntropyLoss()
scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

fases = [
    {"epochs": 10, "lr": 5e-4},
    {"epochs": 10, "lr": 3e-4},
    {"epochs": 5,  "lr": 1e-4},
    {"epochs": 3,  "lr": 5e-5},
]

# ================================================================
# 📈 Función: métricas de diversidad
# ================================================================
def evaluar_texto_generado(texto):
    tokens = texto.split()
    n = len(tokens)
    if n < 3:
        return {"dist2": 0, "dist3": 0, "rep_ngrams": 0, "longitud": n, "vocab": 0}

    bigramas = list(zip(tokens, tokens[1:]))
    trigramas = list(zip(tokens, tokens[1:], tokens[2:]))

    dist2 = len(set(bigramas)) / len(bigramas)
    dist3 = len(set(trigramas)) / len(trigramas)
    rep_ngrams = 1 - dist3  # cuanto más bajo mejor
    vocab_div = len(set(tokens)) / n
    return {
        "dist2": round(dist2, 3),
        "dist3": round(dist3, 3),
        "rep_ngrams": round(rep_ngrams, 3),
        "longitud": n,
        "vocab": round(vocab_div, 3)
    }

# ================================================================
# 🚀 Entrenamiento progresivo
# ================================================================
for i, fase in enumerate(fases, start=1):
    lr = fase["lr"]
    epochs = fase["epochs"]
    steps_per_epoch = (len(data_train) - 1) // (seq_len * batch_size * accum_steps)
    total_steps = steps_per_epoch * epochs

    optimizador = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizador, max_lr=lr, total_steps=total_steps,
        anneal_strategy='cos', pct_start=0.1
    )

    print(f"\n--- 🧭 Fase {i} | lr={lr} | epochs={epochs} ---")

    step_count = 0
    for epoch in range(1, epochs + 1):
        modelo.train()
        perdida_total, accuracy_total, num_batches = 0, 0, 0

        for i_batch, (x_batch, y_batch) in enumerate(crear_batches(data_train, seq_len, batch_size, device)):
            with amp.autocast("cuda"):
                salida = modelo(x_batch)
                perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1)) / accum_steps

            if scaler:
                scaler.scale(perdida).backward()
            else:
                perdida.backward()

            if (i_batch + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizador)
                    scaler.update()
                else:
                    optimizador.step()
                optimizador.zero_grad()

                # ✅ Solo step mientras no exceda total_steps
                if step_count < total_steps:
                    scheduler.step()
                    step_count += 1

            pred = salida.argmax(dim=-1)
            acc = (pred == y_batch).float().mean().item()
            perdida_total += perdida.item() * accum_steps
            accuracy_total += acc
            num_batches += 1

        # --- 🔍 Validación ---
        modelo.eval()
        with torch.no_grad():
            perdida_val, accuracy_val = 0, 0
            num_batches_val = 0
            for x_batch, y_batch in crear_batches(data_val, seq_len, batch_size, device):
                salida_val = modelo(x_batch)
                perdida_batch = criterio(salida_val.view(-1, vocab_size), y_batch.view(-1)).item()
                acc_val = (salida_val.argmax(dim=-1) == y_batch).float().mean().item()
                perdida_val += perdida_batch
                accuracy_val += acc_val
                num_batches_val += 1

        perdida_val_media = perdida_val / num_batches_val
        accuracy_val_media = accuracy_val / num_batches_val
        perplexity = torch.exp(torch.tensor(perdida_val_media))
        lr_actual = optimizador.param_groups[0]["lr"]

        print(f"Fase {i} - Epoch {epoch}/{epochs} | "
              f"Train: loss={perdida_total/num_batches:.4f}, acc={accuracy_total/num_batches:.4f} | "
              f"Val: loss={perdida_val_media:.4f}, acc={accuracy_val_media:.4f}, ppl={perplexity:.2f} | "
              f"LR={lr_actual:.6f}")

        # --- 💾 Checkpoint + métricas de texto ---
        if epoch % checkpoint_every == 0:
            checkpoint_data = {
                "modelo": modelo.state_dict(),
                "fase": i, "epoch": epoch
            }
            torch.save(checkpoint_data, ruta_modelo_drive)
            print(f"💾 Checkpoint guardado en Drive tras epoch {epoch}")

            texto_prueba = generar_texto(
                modelo=modelo, texto_inicio="el arte", longitud=120,
                temperatura=0.9, seq_len=seq_len, device=device,
                tokenizer=tokenizer, top_k=50
            )
            metricas = evaluar_texto_generado(texto_prueba)
            print(f"\n💬 Texto de prueba tras epoch {epoch}:\n{texto_prueba}\n")
            print(f"📊 Métricas de texto: {metricas}\n")

print("✅ Entrenamiento finalizado y modelo guardado:", ruta_modelo_drive)
