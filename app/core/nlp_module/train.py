# app/core/nlp_module/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import crear_batches, codificar
from transformer import Transformer

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 100       # aumenta la longitud de secuencia para mejor coherencia
batch_size = 16     # ajusta según memoria GPU
epochs = 20
lr = 0.001

# Ruta al dataset
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/dataset_completo.txt")
print("Leyendo dataset en:", ruta_dataset)

# Leer y preprocesar dataset completo
with open(ruta_dataset, "r", encoding="utf-8") as f:
    texto = f.read().lower()

print(f"Dataset completo: {len(texto)} caracteres.")

# Codificar corpus
data = [codificar(texto)]

# Ajustar para que cada secuencia tenga exactamente seq_len
total_len = len(data[0])
ajuste = (total_len - 1) % seq_len  # -1 por la ventana objetivo
if ajuste != 0:
    print(f"Recortando {ajuste} caracteres para que todas las secuencias tengan longitud {seq_len}")
    data[0] = data[0][:-ajuste]

# Tamaño del vocabulario
vocab_size = len(sorted(list("abcdefghijklmnopqrstuvwxyzáéíóúü ,.!?\n")))

# Inicializar modelo
modelo = Transformer(vocab_size=vocab_size).to(device)
optimizador = optim.Adam(modelo.parameters(), lr=lr)
criterio = nn.CrossEntropyLoss()

# Entrenamiento
for epoch in range(1, epochs+1):
    modelo.train()
    perdida_total = 0
    for x_batch, y_batch in crear_batches(data, seq_len, batch_size, device):
        optimizador.zero_grad()
        salida = modelo(x_batch)
        perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)  # más estable
        optimizador.step()
        perdida_total += perdida.item()
    if epoch % 5 == 0 or epoch == 1:  # imprime más seguido en dataset grande
        print(f"Epoch {epoch}/{epochs} - Pérdida: {perdida_total:.4f}")

# Guardar modelo entrenado
torch.save(modelo.state_dict(), "transformer_art_model.pth")
print("Modelo guardado como transformer_art_model.pth")
