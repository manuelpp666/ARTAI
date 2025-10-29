# train_colab_progresivo.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import construir_vocab, guardar_vocab, codificar, crear_batches
from transformer import Transformer

# ----------------------
# Configuración general
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 50
batch_size = 8
checkpoint_every = 2  # ✅ Guardar cada 2 epochs

# ----------------------
# Rutas
# ----------------------
ruta_dataset = os.path.join(os.path.dirname(__file__), "../../../datasets/español/arte_traducido/dataset_completo.txt")
ruta_vocab = os.path.join(os.path.dirname(__file__),"../../../models/vocab_art.pt")
ruta_modelo_local = os.path.join(os.path.dirname(__file__), "../../../models/transformer_art_model.pth")
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
# Crear vocabulario dinámico
# ----------------------
chars, stoi, itos = construir_vocab(texto)
guardar_vocab(stoi, itos, ruta_vocab)
vocab_size = len(chars)
print(f"✅ Vocabulario construido: {vocab_size} caracteres únicos")

# ----------------------
# Preparar datos
# ----------------------
data = [codificar(texto, stoi)]
total_len = len(data[0])
ajuste = (total_len - 1) % seq_len
if ajuste != 0:
    print(f"Recortando {ajuste} caracteres para que todas las secuencias tengan longitud {seq_len}")
    data[0] = data[0][:-ajuste]

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
# Cargar pesos iniciales
# ----------------------
inicio_fase = 0
inicio_epoch = 1
optimizador = None

# ----------------------
# Reanudar desde checkpoint si existe
# ----------------------
if os.path.exists(ruta_modelo_drive):
    print("✅ Cargando checkpoint previo desde Drive:", ruta_modelo_drive)
    checkpoint = torch.load(ruta_modelo_drive, map_location=device)
    modelo.load_state_dict(checkpoint["modelo"])
    optimizador = optim.Adam(modelo.parameters(), lr=fases[checkpoint["fase"]]["lr"])
    if "optimizador" in checkpoint:
        optimizador.load_state_dict(checkpoint["optimizador"])
    inicio_fase = checkpoint["fase"]
    inicio_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Reanudando entrenamiento desde Fase {inicio_fase+1}, Epoch {inicio_epoch}")
elif os.path.exists(ruta_modelo_local):
    print("✅ Cargando modelo base local desde:", ruta_modelo_local)
    checkpoint = torch.load(ruta_modelo_local, map_location=device)
    modelo_es_dict = checkpoint["modelo"] if "modelo" in checkpoint else checkpoint

    # Transfer learning de embeddings con diferente vocab_size
    with torch.no_grad():
        embedding_es = modelo_es_dict["embedding.weight"]
        vocab_antiguo = embedding_es.shape[0]
        vocab_nuevo = modelo.embedding.weight.shape[0]

        # Copiar los embeddings existentes (hasta donde se pueda)
        min_vocab = min(vocab_antiguo, vocab_nuevo)
        modelo.embedding.weight[:min_vocab] = embedding_es[:min_vocab]

        # Inicializar los nuevos símbolos aleatoriamente
        if vocab_nuevo > vocab_antiguo:
            nn.init.normal_(modelo.embedding.weight[vocab_antiguo:], mean=0.0, std=0.02)

    print(f"✅ Embeddings transferidos ({min_vocab} comunes, {vocab_nuevo - vocab_antiguo} nuevos inicializados)")
else:
    print("⚠️ No se encontró modelo local ni checkpoint. Entrenamiento desde cero.")

# ----------------------
# Entrenamiento por fases
# ----------------------
for i, fase in enumerate(fases[inicio_fase:], start=inicio_fase):
    print(f"\n--- Fase {i+1} | lr={fase['lr']} | epochs={fase['epochs']} ---")
    optimizador = optim.Adam(modelo.parameters(), lr=fase["lr"])
    
    for epoch in range(inicio_epoch, fase["epochs"] + 1):
        modelo.train()
        perdida_total = 0
        num_batches = 0 
        
        for x_batch, y_batch in crear_batches(data, seq_len, batch_size, device):
            optimizador.zero_grad()
            salida = modelo(x_batch)
            perdida = criterio(salida.view(-1, vocab_size), y_batch.view(-1))
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            perdida_total += perdida.item()
            num_batches += 1
        
        perdida_media = perdida_total / num_batches
        print(f"Fase {i+1} - Epoch {epoch}/{fase['epochs']} - Pérdida media: {perdida_media:.4f}")
        print(f"Fase {i+1} - Epoch {epoch}/{fase['epochs']} - Pérdida: {perdida_total:.4f}")


        # 💾 Guardar checkpoint cada 2 epochs
        if epoch % checkpoint_every == 0:
            checkpoint_data = {
                "modelo": modelo.state_dict(),
                "optimizador": optimizador.state_dict(),
                "fase": i,
                "epoch": epoch
            }
            torch.save(checkpoint_data, ruta_modelo_drive)
            print(f"💾 Checkpoint guardado en Drive después de epoch {epoch}")

    # Reiniciar contador de epoch para la siguiente fase
    inicio_epoch = 1

# ----------------------
# Guardar modelo final
# ----------------------
torch.save({"modelo": modelo.state_dict()}, ruta_modelo_drive)
print("✅ Entrenamiento finalizado. Modelo guardado en Drive:", ruta_modelo_drive)
