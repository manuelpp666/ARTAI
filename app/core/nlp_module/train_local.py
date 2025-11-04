# ================================================================
# train_local.py — versión local optimizada (sin Drive, sin checkpoints intermedios)
# ================================================================
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from preprocess import construir_vocab, guardar_vocab, codificar, generar_batches
from transformer import Transformer
from generator import generar_texto
from torch import amp
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm  # <--- NUEVO: Para la barra de progreso


# -------------------------------------------------
# 🔄 Scheduler corregido con warmup normalizado
# -------------------------------------------------
def make_lr_lambda(warmup_steps):
    def lr_lambda(step):
        step = max(1, step + 1)
        if step <= warmup_steps:
            return float(step) / float(warmup_steps)
        return (step ** -0.5) * (warmup_steps ** 0.5)
    return lr_lambda


# ----------------------
# 🔧 Configuración general
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 1024
batch_size = 4
accum_steps = 4
porc_validacion = 0.1
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ----------------------
# 📁 Rutas locales
# ----------------------
base_dir = os.path.dirname(__file__)
ruta_dataset = os.path.join(base_dir, "../../../datasets/español/arte_traducido/dataset_completo.txt")
ruta_modelo_final = os.path.join(base_dir, "../../../models/transformer_art_model_final.pth")
ruta_checkpoint_final = ruta_modelo_final.replace(".pth", "_checkpoint.pth")
os.makedirs(os.path.dirname(ruta_modelo_final), exist_ok=True)

# ----------------------
# 📖 Cargar dataset
# ----------------------
print(f"Leyendo dataset desde: {ruta_dataset}")
with open(ruta_dataset, "r", encoding="utf-8") as f:
    lineas = f.readlines()

print("Barajando el dataset para una validación robusta...")
random.shuffle(lineas) # <--- MODIFICADO (ANTI-OVERFITTING)

num_train_lines = int(len(lineas) * (1 - porc_validacion))
lineas_train = lineas[:num_train_lines]
lineas_val = lineas[num_train_lines:]
print(f"Entrenamiento: {len(lineas_train)} líneas | Validación: {len(lineas_val)} líneas")

# ----------------------
# 🧩 Crear vocabulario
# ----------------------
tokenizer, stoi, itos = construir_vocab(ruta_dataset, ruta_vocab="bpe_tokenizer.json", vocab_size=10000)
guardar_vocab(stoi, itos, "bpe_tokenizer.json")

if "SECCION" not in stoi:
    raise ValueError("❌ Token especial 'SECCION' no encontrado en vocabulario.")

token_seccion_id = stoi["SECCION"]
vocab_size = len(stoi)
print(f"📘 Vocabulario listo con {vocab_size} tokens")

# ----------------------
# 🔡 Codificar datasets
# ----------------------
# Nota: Los generadores se crearán dentro del bucle de epoch para el shuffling
print("Generadores de batches listos.")


# ----------------------
# 🧠 Inicializar modelo
# ----------------------
print("Inicializando modelo PEQUEÑO y REGULARIZADO para evitar overfitting...")
# <--- MODIFICADO (ANTI-OVERFITTING): Arquitectura más pequeña y con más dropout
modelo = Transformer(
    vocab_size=vocab_size,
    d_model=384,        # Reducido de 512
    N=4,                # Reducido de 5
    num_heads=6,        # Reducido de 8
    d_ff=1536,          # d_model * 4
    max_len=1024,
    dropout=0.35        # Dropout base más alto
).to(device)

criterio = nn.CrossEntropyLoss(label_smoothing=0.05)

# --- Token de reemplazo dinámico ---
token_artista_id = stoi.get("<ARTISTA>")
if token_artista_id is None:
    token_artista_id = len(stoi)
    stoi["<ARTISTA>"] = token_artista_id
    itos[token_artista_id] = "<ARTISTA>"
    vocab_size += 1
    # Re-ajustar capas de embedding y salida
    modelo.embedding = nn.Embedding(vocab_size, modelo.embedding.embedding_dim).to(device)
    modelo.out = nn.Linear(modelo.out.in_features, vocab_size).to(device)


# --- Token masking ---
def aplicar_token_masking(x_batch, token_seccion_id, token_artista_id, prob_mask=0.20): # <--- MODIFICADO (ANTI-OVERFITTING): 0.15 a 0.20
    x_masked = x_batch.clone()
    mask = (x_masked != token_seccion_id) & (torch.rand_like(x_masked.float()) < prob_mask)
    x_masked[mask] = token_artista_id
    return x_masked


# ----------------------
# 🔁 Fases de entrenamiento
# ----------------------
fases = [
    {"epochs": 4,  "lr": 2e-4},
    {"epochs": 12, "lr": 8e-5},
    {"epochs": 8,  "lr": 5e-5},
    {"epochs": 4,  "lr": 2e-5},
]

# ----------------------
# ⚙️ Escalador AMP
# ----------------------
scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

# ----------------------
# 🚀 Entrenamiento principal
# ----------------------
for i, fase in enumerate(fases, start=1):
    print(f"\n--- Fase {i} | LR={fase['lr']} | Epochs={fase['epochs']} ---")

    # <--- MODIFICADO (ANTI-OVERFITTING): weight_decay aumentado
    optimizador = optim.AdamW(modelo.parameters(), lr=fase["lr"], weight_decay=0.05)
    scheduler = LambdaLR(optimizador, lr_lambda=make_lr_lambda(2500))

    for epoch in range(1, fase["epochs"] + 1):
        modelo.train()
        total_loss, total_acc, num_batches = 0, 0, 0

        # Barajar líneas de entrenamiento en cada epoch
        random.shuffle(lineas_train)
        # Crear generador y barra de progreso para esta epoch
        train_gen = generar_batches(lineas_train, tokenizer, seq_len, batch_size, token_seccion_id, device)
        
        # <--- NUEVO: Barra de progreso TQDM para el entrenamiento
        progress_bar = tqdm(train_gen, desc=f"Fase {i} Epoch {epoch}/{fase['epochs']} (Train)", leave=False)

        for i_batch, (x_batch, y_batch) in enumerate(progress_bar):
            with amp.autocast("cuda"):
                # <--- MODIFICADO: Se usará el nuevo default de prob_mask=0.20
                x_masked = aplicar_token_masking(x_batch, token_seccion_id, token_artista_id)
                salida = modelo(x_masked)
                perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1)) / accum_steps

                if scaler:
                    scaler.scale(perdida).backward()
                else:
                    perdida.backward()

            if (i_batch + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizador)
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    scaler.step(optimizador)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    optimizador.step()
                optimizador.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += perdida.item() * accum_steps
            total_acc += (salida.argmax(-1) == y_batch).float().mean().item()
            num_batches += 1
            
            # <--- NUEVO: Actualizar barra de progreso con métricas
            if i_batch % 10 == 0: # Actualizar cada 10 batches
                progress_bar.set_postfix(loss=f"{(total_loss / num_batches):.4f}", acc=f"{(total_acc / num_batches):.4f}")

        # Cerrar barra de progreso de entrenamiento
        progress_bar.close()

        if num_batches == 0:
            print(f"Fase {i} - Epoch {epoch}/{fase['epochs']} | No se procesaron batches de entrenamiento.")
            continue

        loss_train = total_loss / num_batches
        acc_train = total_acc / num_batches
        ppl_train = torch.exp(torch.tensor(loss_train))

        # ------------------ VALIDACIÓN ------------------
        modelo.eval()
        with torch.no_grad():
            loss_val, acc_val, nb_val = 0, 0, 0
            val_gen = generar_batches(lineas_val, tokenizer, seq_len, batch_size, token_seccion_id, device)
            
            # <--- NUEVO: Barra de progreso TQDM para la validación
            val_progress_bar = tqdm(val_gen, desc=f"Fase {i} Epoch {epoch}/{fase['epochs']} (Val)", leave=False)
            
            for x_batch, y_batch in val_progress_bar:
                salida_val = modelo(x_batch)
                loss_val += criterio(salida_val.view(-1, vocab_size), y_batch.view(-1)).item()
                acc_val += (salida_val.argmax(-1) == y_batch).float().mean().item()
                nb_val += 1
            
            val_progress_bar.close()

        if nb_val == 0:
            print(f"Fase {i} - Epoch {epoch}/{fase['epochs']} | No se procesaron batches de validación.")
            continue

        loss_val /= nb_val
        acc_val /= nb_val
        ppl_val = torch.exp(torch.tensor(loss_val))
        lr_actual = optimizador.param_groups[0]["lr"]

        print(f"Fase {i} - Epoch {epoch}/{fase['epochs']} | "
              f"Train loss={loss_train:.4f}, acc={acc_train:.4f}, ppl={ppl_train:.2f} | "
              f"Val loss={loss_val:.4f}, acc={acc_val:.4f}, ppl={ppl_val:.2f} | LR={lr_actual:.6f}")
        
        # <--- NUEVO: Generar texto cada 2 épocas o en la última época de la fase
        if epoch % 2 == 0 or epoch == fase["epochs"]:
            print("--- Generando texto de muestra ---")
            modelo.eval() # Asegurarse de que esté en modo evaluación
            try:
                ejemplo_intermedio = generar_texto(
                    modelo=modelo,
                    tokenizer=tokenizer,
                    device=device,
                    seed_text="SECCION Pablo Picasso SECCION",
                    max_length=320,
                    top_k=40,
                    top_p=0.90,
                    temperature=0.7,
                    repetition_penalty=1.15
                )
                print(f"💬 Muestra (Epoch {epoch}):\n{ejemplo_intermedio}\n")
            except Exception as e:
                print(f"❌ Error al generar texto de muestra: {e}")
            # El modelo se volverá a poner en modo .train() al inicio del siguiente bucle de epoch


# ----------------------
# 💾 Guardado final
# ----------------------
torch.save({"modelo": modelo.state_dict()}, ruta_modelo_final)
print(f"✅ Modelo final guardado en: {ruta_modelo_final}")

checkpoint_final = {
    "modelo": modelo.state_dict(),
    "optimizador": optimizador.state_dict(),
    "scheduler": scheduler.state_dict(),
    "fase": len(fases),
    "epoch": fases[-1]["epochs"],
    "scaler": scaler.state_dict() if scaler else None,
}
torch.save(checkpoint_final, ruta_checkpoint_final)
print(f"💾 Checkpoint final guardado en: {ruta_checkpoint_final}")

# ----------------------
# 🖋️ Generar texto de muestra
# ----------------------
print("--- Generando texto de muestra FINAL ---")
ejemplo = generar_texto(
    modelo=modelo,
    tokenizer=tokenizer,
    device=device,
    seed_text="SECCION Pablo Picasso SECCION",
    max_length=320,
    top_k=40,
    top_p=0.90,                     # <--- MODIFICADO (ANTI-OVERFITTING): Ligeramente más conservador
    temperature=0.7,
    repetition_penalty=1.15         # <--- MODIFICADO (ANTI-OVERFITTING): Reducido para evitar artefactos
)
print(f"\n💬 Texto generado de prueba:\n{ejemplo}\n")