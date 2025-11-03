import torch
from preprocess import construir_vocab
from transformer import Transformer
from generator import generar_texto
import os
# ============================================================
# 🔹 Configuración
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = os.path.join(os.path.dirname(__file__),"/content/ARTAI/models/arte/transformer_arte_model.pth")
dataset_path = os.path.join(os.path.dirname(__file__), "../../../datasets/español/arte_traducido/dataset_completo.txt")

# ============================================================
# 🔹 Cargar o entrenar tokenizer
# ============================================================
tokenizer, stoi, itos = construir_vocab(dataset_path, vocab_size=10000)

# ============================================================
# 🔹 Cargar modelo
# ⚠️ Ajusta los parámetros según tu entrenamiento
# ============================================================
vocab_size = tokenizer.get_vocab_size()
modelo = Transformer(
    vocab_size=vocab_size,
    d_model=384,
    N=4,
    num_heads=6,
    d_ff=1536,
    max_len=512,
    dropout=0.1
)

# Cargar checkpoint si existe
try:
    modelo.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Checkpoint cargado desde {checkpoint_path}")
except FileNotFoundError:
    print("⚠️ No se encontró checkpoint, el modelo estará inicializado desde cero")

modelo.to(device)
modelo.eval()

# ============================================================
# 🔹 Generar texto de prueba
# ============================================================
seed_text = "SECCION Pablo Picasso SECCION"
try:
    texto_generado = generar_texto(
        modelo,
        tokenizer,
        device,
        seed_text,
        max_length=200,
        top_k=40,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.15
    )
    print("✅ Texto generado con éxito:\n")
    print(texto_generado[:1000], "\n...")  # primeros 1000 caracteres
except Exception as e:
    print("❌ Error durante la generación:")
    print(e)
