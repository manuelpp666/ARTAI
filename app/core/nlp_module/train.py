# ================================================================
# train_colab_progresivo.py — versión optimizada para dataset de 100 MB
# ================================================================
import os
import math
import re
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import construir_vocab, guardar_vocab, codificar, generar_batches
from transformer import Transformer
from generator import generar_texto
from torch import amp  # ✅ NUEVO: reemplaza torch.cuda.amp
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(step):
    warmup_steps = 2000
    return min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))

# ----------------------
# 🔧 Ajustes de rendimiento GPU
# ----------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ----------------------
# Configuración general
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 320             # ✅ Contexto mayor, ideal para textos largos
batch_size = 8
accum_steps = 2             # ✅ Gradient accumulation
checkpoint_every = 2        # ✅ Guardar cada 2 epochs
porc_validacion = 0.1       # ✅ 10% para validación

# ----------------------
# Rutas
# ----------------------
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/arte_traducido/dataset_completo.txt")
ruta_modelo_local = os.path.join(os.path.dirname(__file__), "../../../models/transformer_art_model.pth")
ruta_modelo_drive = "/content/drive/MyDrive/arte_chatbot/models/transformer_art_model.pth"
os.makedirs(os.path.dirname(ruta_modelo_drive), exist_ok=True)

# ----------------------
# Leer dataset
# ----------------------
print("Leyendo dataset en:", ruta_dataset)
with open(ruta_dataset, "r", encoding="utf-8") as f:
    lineas = f.readlines()

num_train_lines = int(len(lineas) * (1 - porc_validacion))
lineas_train = lineas[:num_train_lines]
lineas_val   = lineas[num_train_lines:]
print(f"Entrenamiento: {len(lineas_train)} chars | Validación: {len(lineas_val)} chars (corte limpio)")

# ----------------------
# Crear vocabulario dinámico
# ----------------------
tokenizer, stoi, itos = construir_vocab(ruta_dataset, ruta_vocab="bpe_tokenizer.json", vocab_size=10000)
guardar_vocab(stoi, itos, "bpe_tokenizer.json")
if "SECCION" not in stoi:
    raise ValueError("❌ Token especial 'SECCION' no encontrado en vocabulario.")
token_seccion_id = stoi["SECCION"]
print(f"✅ Token 'SECCION' ID: {token_seccion_id}")
vocab_size = len(stoi)
print(f"📘 Vocabulario con {vocab_size} tokens")

# ----------------------
# Codificar datasets
# ----------------------
data_train = generar_batches(lineas_train, tokenizer, seq_len, batch_size, token_seccion_id, device)
data_val   = generar_batches(lineas_val, tokenizer, seq_len, batch_size, token_seccion_id, device)


# ----------------------
# Inicializar modelo y criterio
# ----------------------
modelo = Transformer(vocab_size=vocab_size).to(device)
criterio = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------------
# Definir fases de entrenamiento
# ----------------------
fases = [
    {"epochs": 12, "lr": 2e-4},
    {"epochs": 10, "lr": 1e-4},
    {"epochs": 5,  "lr": 1e-4},
    {"epochs": 3,  "lr": 5e-5},
]

# ----------------------
# Cargar pesos iniciales
# ----------------------
inicio_fase = 0
inicio_epoch = 1
optimizador = None
scheduler = None


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

if os.path.exists(ruta_modelo_drive):
    print("✅ Cargando checkpoint previo desde Drive:", ruta_modelo_drive)
    checkpoint = torch.load(ruta_modelo_drive, map_location=device)
    modelo.load_state_dict(checkpoint["modelo"])
    inicio_fase = checkpoint["fase"]
    inicio_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Reanudando desde Fase {inicio_fase+1}, Epoch {inicio_epoch}")
else:
    print("⚠️ No se encontró modelo local ni checkpoint. Entrenamiento desde cero.")


# ----------------------
# Escalador AMP
# ----------------------
scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

# ----------------------
# Entrenamiento
# ----------------------
for i, fase in enumerate(fases[inicio_fase:], start=inicio_fase):
    print(f"\n--- Fase {i+1} | LR={fase['lr']} | Epochs={fase['epochs']} ---")
    
    optimizador = optim.AdamW(modelo.parameters(), lr=fase["lr"], weight_decay=0.01)
    scheduler = LambdaLR(optimizador, lr_lambda)    


    for epoch in range(inicio_epoch, fase["epochs"] + 1):
        modelo.train()
        perdida_total = 0
        accuracy_total = 0
        num_batches = 0

        train_gen = generar_batches(lineas_train, tokenizer, seq_len, batch_size, token_seccion_id, device)
        for i_batch, (x_batch, y_batch) in enumerate(train_gen):            # ✅ NUEVO: autocast actualizado
            with amp.autocast("cuda"):
                salida = modelo(x_batch)
                perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1))
                perdida = perdida / accum_steps

                if scaler is not None:
                    scaler.scale(perdida).backward()
                else:
                    perdida.backward()

            if (i_batch + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizador)
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    scaler.step(optimizador)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    optimizador.step()
                    
                
                optimizador.zero_grad(set_to_none=True)
                
                
                scheduler.step()

            pred = salida.argmax(dim=-1)
            acc = (pred == y_batch).float().mean().item()
            perdida_total += perdida.item() * accum_steps
            accuracy_total += acc
            num_batches += 1

        perdida_media = perdida_total / num_batches
        accuracy_media = accuracy_total / num_batches

        # ----------------------
        # Validación
        # ----------------------
        modelo.eval()
        with torch.no_grad():
            perdida_val = 0
            accuracy_val = 0
            num_batches_val = 0
            val_gen = generar_batches(lineas_val, tokenizer, seq_len, batch_size, token_seccion_id, device)
            for x_batch, y_batch in val_gen:
                salida_val = modelo(x_batch)
                perdida_batch = criterio(salida_val.view(-1, vocab_size), y_batch.view(-1)).item()
                pred_val = salida_val.argmax(dim=-1)
                acc_val = (pred_val == y_batch).float().mean().item()
                perdida_val += perdida_batch
                accuracy_val += acc_val
                num_batches_val += 1

            perdida_val_media = perdida_val / num_batches_val
            accuracy_val_media = accuracy_val / num_batches_val
            perplexity = torch.exp(torch.tensor(perdida_val_media))
        lr_actual = optimizador.param_groups[0]["lr"]
        
        print(f"Fase {i+1} - Epoch {epoch}/{fase['epochs']} | "
              f"Train: loss={perdida_media:.4f}, acc={accuracy_media:.4f} | "
              f"Val: loss={perdida_val_media:.4f}, acc={accuracy_val_media:.4f}, ppl={perplexity:.2f} | "
              f"LR={lr_actual:.6f}")

        # ----------------------
        # Checkpoint
        # ----------------------
        if epoch % checkpoint_every == 0:
            checkpoint_data = {
                "modelo": modelo.state_dict(),
                "optimizador": optimizador.state_dict(),
                "scheduler": scheduler.state_dict(),  # <-- guardar scheduler
                "fase": i,
                "epoch": epoch,
            }
            torch.save(checkpoint_data, ruta_modelo_drive)
            print(f"💾 Checkpoint guardado en Drive después de epoch {epoch}")

            ejemplo = generar_texto(
                modelo=modelo,
                tokenizer=tokenizer,
                device=device,
                seed_text="Qué es el arte?",       # texto inicial
                max_length=320,            # longitud de generación
                top_k=80,                  # top-k sampling
                top_p=0.95,                 # nucleus sampling
                temperature=0.8,           # suaviza la probabilidad
                repetition_penalty=1.2     # penalización de repetición
            )
            
            metricas = evaluar_texto_generado(ejemplo)
            print(f"\n💬 Texto de prueba tras epoch {epoch}:\n{ejemplo}\n")
            print(f"📊 Métricas de texto: {metricas}\n")

    inicio_epoch = 1

# ----------------------
# Guardar modelo final
# ----------------------
torch.save({"modelo": modelo.state_dict()}, ruta_modelo_drive)
print("✅ Entrenamiento finalizado. Modelo guardado en Drive:", ruta_modelo_drive)
