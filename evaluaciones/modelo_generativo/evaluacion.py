# ============================================================
# evaluacion_clip_cpu.py — Evaluación CLIP optimizada para CPU
# ============================================================

import os
import json
import time
import gc
import requests
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from gradio_client import Client
from transformers import CLIPProcessor, CLIPModel

# ============================================
# CONFIGURACIÓN
# ============================================
HF_REPO_ID = "Joseph1112/ArtAI"     # Tu Space / modelo en Hugging Face
PROMPTS_FILE = "prompts_365.jsonl"  # Archivo con prompts
RESULTS_FOLDER = "resultados_eval"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

PARTIAL_SAVE_EVERY = 10  # 💾 Guarda cada 10 resultados (para CPU)
SLEEP_BETWEEN_CALLS = 3  # 😴 Pausa entre llamadas (segundos, evita saturar red)
RESULTS_PATH = os.path.join(RESULTS_FOLDER, "clipscore_resultados.jsonl")

# ============================================
# 1. CLIENTE GRADIO
# ============================================
client = Client(HF_REPO_ID)

# ============================================
# 2. MODELO CLIP (optimizado para CPU)
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = (
    CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    .to(device)
    .eval()  # evita gradientes innecesarios
)
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

print(f"🔹 Dispositivo usado: {device.upper()}")
print(f"🔹 CLIP cargado correctamente.\n")

# ============================================
# 3. CARGAR PROGRESO ANTERIOR
# ============================================
procesados = set()
resultados = []

if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            resultados.append(data)
            procesados.add(data["prompt"])

print(f"✅ Se cargaron {len(procesados)} resultados previos.\n")

# ============================================
# 4. FUNCIONES AUXILIARES
# ============================================

def generar_imagen(prompt):
    """Llama al modelo en Hugging Face para generar una imagen."""
    try:
        result = client.predict(prompt=prompt, api_name="/predict")

        img_path = os.path.join(RESULTS_FOLDER, f"{hash(prompt)}.png")

        # Puede retornar URL o ruta local
        if isinstance(result, str) and result.startswith("http"):
            response = requests.get(result)
            with open(img_path, "wb") as f:
                f.write(response.content)
        else:
            Image.open(result).save(img_path)

        return img_path

    except Exception as e:
        print(f"❌ Error generando imagen para '{prompt}': {e}")
        return None


def calcular_clip_score(prompt, image_path):
    """Calcula CLIPScore entre prompt y la imagen generada."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            score = outputs.logits_per_image.item()

        return score

    except Exception as e:
        print(f"⚠️ Error en CLIPScore para '{prompt}': {e}")
        return None


def guardar_parcial(resultados):
    """Guarda progreso incremental."""
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 Progreso guardado ({len(resultados)} resultados).")

# ============================================
# 5. LOOP PRINCIPAL (con control de CPU y red)
# ============================================
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

pendientes = [p for p in prompts if p not in procesados]
print(f"Evaluando {len(pendientes)} prompts nuevos...\n")

# 💡 Puedes limitar para pruebas
# pendientes = pendientes[:30]

for i, prompt in enumerate(tqdm(pendientes, desc="Evaluando CLIPScore")):
    img_path = generar_imagen(prompt)
    if img_path:
        score = calcular_clip_score(prompt, img_path)
        if score is not None:
            resultados.append({"prompt": prompt, "clipscore": score})

    # 🔹 Limpieza de memoria cada iteración
    gc.collect()

    # 🔹 Guardar cada N iteraciones
    if (i + 1) % PARTIAL_SAVE_EVERY == 0:
        guardar_parcial(resultados)

    # 🔹 Esperar un poco para no sobrecargar CPU/red
    time.sleep(SLEEP_BETWEEN_CALLS)

# Guardar al final
guardar_parcial(resultados)

# ============================================
# 6. ANÁLISIS Y GRÁFICOS
# ============================================
scores = [r["clipscore"] for r in resultados if r.get("clipscore") is not None]

if len(scores) > 0:
    print("\n=== ESTADÍSTICAS CLIPScore ===")
    print(f"Promedio: {np.mean(scores):.4f}")
    print(f"Desviación estándar: {np.std(scores):.4f}")
    print(f"Mínimo: {np.min(scores):.4f}")
    print(f"Máximo: {np.max(scores):.4f}")
    print(f"Percentil 25/50/75: {np.percentile(scores, [25, 50, 75])}")

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribución del CLIPScore (calidad semántica imagen ↔ prompt)")
    plt.xlabel("CLIPScore")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_FOLDER, "clipscore_histograma.png"))
    plt.show()
else:
    print("⚠️ No se generaron resultados para analizar.")
