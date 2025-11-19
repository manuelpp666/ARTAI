# ================================================================
# main.py — Interfaz Flask para ArtAI (Imagen Local + Texto)
# ================================================================
from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image

# ---------------------------
# Librerías para Difusión Local (Imagen)
# ---------------------------
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel

# ---------------------------
# Modelos de Texto internos
# ---------------------------
from models_utils.arte_desde_cero import cargar_modelo_desde_cero, generar_texto
from models_utils.arte_loader import get_qa

# ---------------------------
# Inicialización Flask
# ---------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Configuración de dispositivo (igual que en test_generator.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"⚙️  Corriendo en: {device}")


# ================================================================
# 1️⃣ CARGAR MODELO DE DIFUSIÓN (IMAGEN) - LOCAL
# ================================================================
print("\n🎨 Cargando modelo de generación de imágenes (Local)...")

# Rutas (Asegúrate que esta carpeta exista y tenga tu modelo entrenado)
RUTA_UNET_ENTRENADA = "models/diffusion_art_model"
MODELO_BASE_ID = "CompVis/stable-diffusion-v1-4"

pipeline_difusion = None

try:
    # 1. Cargar VAE (mejora colores/calidad)
    vae = AutoencoderKL.from_pretrained(MODELO_BASE_ID, subfolder="vae", torch_dtype=torch_dtype)
    
    # 2. Cargar tu UNet entrenada
    unet = UNet2DConditionModel.from_pretrained(RUTA_UNET_ENTRENADA, torch_dtype=torch_dtype)
    
    # 3. Ensamblar Pipeline
    pipeline_difusion = DiffusionPipeline.from_pretrained(
        MODELO_BASE_ID,
        vae=vae,
        unet=unet,
        torch_dtype=torch_dtype,
        safety_checker=None  # Desactivamos checker para evitar errores de memoria o falsos positivos
    )
    
    # Mover a GPU si es posible
    pipeline_difusion = pipeline_difusion.to(device)
    
    # Optimización opcional para ahorrar memoria
    # pipeline_difusion.enable_attention_slicing() 
    
    print(f"✅ Modelo de Difusión cargado correctamente desde: {RUTA_UNET_ENTRENADA}")

except Exception as e:
    print(f"❌ ERROR al cargar modelo de difusión: {e}")
    print("⚠️  El modo 'imagen' no funcionará hasta arreglar la ruta o el modelo.")


# ================================================================
# 2️⃣ CARGAR MODELO LSTM DESDE CERO
# ================================================================
MODEL_PATH_TEXTO = "models/arte/entrenamiento_desde_cero/v_cero.pth"
modelo_cero, word2idx, idx2word = None, None, None

try:
    modelo_cero, word2idx, idx2word = cargar_modelo_desde_cero(MODEL_PATH_TEXTO)
    print("✅ Modelo de texto (LSTM desde cero) cargado.")
except Exception as e:
    print(f"⚠️  No se pudo cargar el modelo desde cero: {e}")


# ================================================================
# 3️⃣ CARGAR MODELO PHI-3 + FAISS (PREENTRENADO)
# ================================================================
print("🧠 Cargando QA preentrenado + FAISS...")
try:
    qa = get_qa()
    print("✅ Modelo Phi-3 + FAISS listo.")
except Exception as e:
    qa = None
    print(f"⚠️  Error en QA Preentrenado: {e}")


# ================================================================
# 4️⃣ RUTAS FLASK
# ================================================================
@app.route('/')
def home():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    modo = data.get("modo")
    mensaje = data.get("message", "")

    # ------------------------------------------------------------
    # MODO IMAGEN (LOCAL)
    # ------------------------------------------------------------
    if modo == "imagen":
        if pipeline_difusion is None:
            return jsonify({"error": "El modelo de generación de imágenes no está cargado."}), 500

        try:
            print(f"🎨 [Difusión Local] Generando imagen para: '{mensaje}'")
            
            # Generar imagen (Inferencia)
            with torch.no_grad():
                output = pipeline_difusion(mensaje, num_inference_steps=50)
                imagen_generada = output.images[0]

            # Guardar imagen
            img_folder = os.path.join(app.static_folder, "generated")
            os.makedirs(img_folder, exist_ok=True)
            
            filename = f"arte_{hash(mensaje)}.png"
            img_path = os.path.join(img_folder, filename)
            
            imagen_generada.save(img_path)

            # Responder con URL
            img_url = f"/static/generated/{filename}"
            return jsonify({"tipo": "imagen", "url": img_url})

        except Exception as e:
            print(f"❌ Error generando imagen: {e}")
            # Limpiar memoria por si acaso
            torch.cuda.empty_cache()
            return jsonify({"error": f"Error interno al generar imagen: {str(e)}"}), 500

    # ------------------------------------------------------------
    # MODO TEXTO LSTM DESDE CERO
    # ------------------------------------------------------------
    elif modo == "texto_cero":
        if modelo_cero is None:
            return jsonify({"tipo": "texto", "texto": "⚠️ El modelo desde cero no está disponible."})

        try:
            respuesta = generar_texto(modelo_cero, word2idx, idx2word, mensaje, max_len=60)
            return jsonify({"tipo": "texto", "texto": respuesta})

        except Exception as e:
            return jsonify({"error": f"Error: {str(e)}"}), 500

    # ------------------------------------------------------------
    # MODO TEXTO PREENTRENADO (PHI-3 + FAISS)
    # ------------------------------------------------------------
    elif modo == "texto_pre":
        if qa is None:
            return jsonify({"tipo": "texto", "texto": "⚠️ El sistema RAG no está disponible."})

        try:
            print(f"🧠 Consultando RAG: {mensaje}")
            # Usamos el wrapper compatible que creamos antes
            result = qa.invoke({"query": mensaje})
            respuesta = result["result"]

            return jsonify({"tipo": "texto", "texto": respuesta})

        except Exception as e:
            return jsonify({"error": f"Error: {str(e)}"}), 500

    else:
        return jsonify({"error": f"Modo desconocido: {modo}"}), 400


# ================================================================
# 5️⃣ EJECUTAR APP
# ================================================================
if __name__ == '__main__':
    # Threaded=False es importante para evitar conflictos con modelos en GPU/Memoria
    app.run(debug=False, port=5000, threaded=False)