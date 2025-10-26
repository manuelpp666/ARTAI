# train_colab_progresivo.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import crear_batches, codificar
from transformer import Transformer

# ----------------------
# Configuración general
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 50
batch_size = 8
checkpoint_every = 5

# ----------------------
# Rutas
# ----------------------
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/dataset_completo.txt")
ruta_modelo_drive = "/content/drive/MyDrive/arte_chatbot/models/transformer_art_model.pth"
os.makedirs(os.path.dirname(ruta_modelo_drive), exist_ok=True)

# ----------------------
# Leer dataset
# ----------------------
print("Leyendo dataset en:", ruta_dataset)
with open(ruta_dataset, "r", encoding="utf-8") as f:
    texto = f.read().lower()
print(f"Dataset completo: {len(texto)} caracteres.")

# ----------------------
# Preparar datos
# ----------------------
data = [codificar(texto)]
total_len = len(data[0])
ajuste = (total_len - 1) % seq_len
if ajuste != 0:
    print(f"Recortando {ajuste} caracteres para que todas las secuencias tengan longitud {seq_len}")
    data[0] = data[0][:-ajuste]

vocab_size = len(sorted(list("abcdefghijklmnopqrstuvwxyzáéíóúü ,.!?\n")))

# ----------------------
# Inicializar modelo y criterio
# ----------------------
modelo = Transformer(vocab_size=vocab_size).to(device)
criterio = nn.CrossEntropyLoss()

# ----------------------
# Definir fases de entrenamiento
# ----------------------
fases = [
    {"epochs": 20, "lr": 0.001},
    {"epochs": 20, "lr": 0.0005},
    {"epochs": 10, "lr": 0.0002},
]

# ----------------------
# Cargar checkpoint si existe
# ----------------------
inicio_fase = 0
inicio_epoch = 1
optimizador = None

if os.path.exists(ruta_modelo_drive):
    print("Cargando checkpoint...")
    checkpoint = torch.load(ruta_modelo_drive, map_location=device)
    if isinstance(checkpoint, dict) and "modelo" in checkpoint:
        modelo.load_state_dict(checkpoint["modelo"])
        inicio_fase = checkpoint.get("fase", 0)
        inicio_epoch = checkpoint.get("epoch", 1) + 1
        print(f"Continuando desde fase {inicio_fase + 1}, epoch {inicio_epoch}")
    else:
        modelo.load_state_dict(checkpoint)
        print("Checkpoint cargado (solo pesos del modelo). Entrenamiento continuará desde fase 1.")

# ----------------------
# Entrenamiento por fases
# ----------------------
for i, fase in enumerate(fases[inicio_fase:], start=inicio_fase):
    print(f"\n--- Fase {i+1} | lr={fase['lr']} | epochs={fase['epochs']} ---")
    optimizador = optim.Adam(modelo.parameters(), lr=fase["lr"])
    
    for epoch in range(inicio_epoch, fase["epochs"] + 1):
        modelo.train()
        perdida_total = 0
        for x_batch, y_batch in crear_batches(data, seq_len, batch_size, device):
            optimizador.zero_grad()
            salida = modelo(x_batch)
            perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1))
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            perdida_total += perdida.item()
        
        print(f"Fase {i+1} - Epoch {epoch}/{fase['epochs']} - Pérdida: {perdida_total:.4f}")

        # Guardar checkpoint cada N epochs
        if epoch % checkpoint_every == 0:
            checkpoint_data = {
                "modelo": modelo.state_dict(),
                "optimizador": optimizador.state_dict(),
                "fase": i,
                "epoch": epoch
            }
            torch.save(checkpoint_data, ruta_modelo_drive)
            print(f"Checkpoint guardado después de epoch {epoch}")

    # Reiniciar contador de epoch para la siguiente fase
    inicio_epoch = 1

# ----------------------
# Guardar modelo final
# ----------------------
torch.save({"modelo": modelo.state_dict()}, ruta_modelo_drive)
print("Entrenamiento finalizado. Modelo guardado en Drive:", ruta_modelo_drive)
