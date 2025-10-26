# train_colab.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import crear_batches, codificar
from transformer import Transformer

# ----------------------
# Configuración
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 150        # longitud de secuencia para coherencia
batch_size = 16      # ajusta según memoria GPU
epochs = 40
lr = 0.001
checkpoint_every = 5  # guardar cada 5 epochs

# ----------------------
# Rutas
# ----------------------
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/dataset_completo.txt")
ruta_modelo_drive = "/content/drive/MyDrive/arte_chatbot/models/transformer_art_model.pth"
os.makedirs(os.path.dirname(ruta_modelo_drive), exist_ok=True)

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
# Inicializar modelo
# ----------------------
modelo = Transformer(vocab_size=vocab_size).to(device)
optimizador = optim.Adam(modelo.parameters(), lr=lr)
criterio = nn.CrossEntropyLoss()

# Cargar checkpoint si existe
if os.path.exists(ruta_modelo_drive):
    modelo.load_state_dict(torch.load(ruta_modelo_drive))
    print("Checkpoint cargado, continuando entrenamiento...")

# ----------------------
# Entrenamiento
# ----------------------
for epoch in range(1, epochs + 1):
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
    
    print(f"Epoch {epoch}/{epochs} - Pérdida: {perdida_total:.4f}")

    # Guardar checkpoint cada N epochs
    if epoch % checkpoint_every == 0:
        torch.save(modelo.state_dict(), ruta_modelo_drive)
        print(f"Checkpoint guardado en Drive después de epoch {epoch}")

# Guardar modelo final
torch.save(modelo.state_dict(), ruta_modelo_drive)
print("Entrenamiento finalizado. Modelo guardado en Drive:", ruta_modelo_drive)
