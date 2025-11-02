import os
from flask import Flask, render_template, request, jsonify, url_for
import torch

# --- Importaciones solo para el Modelo de Difusión ---
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# ---------------------------------------------------
# CONFIGURACIÓN FLASK
# ---------------------------------------------------
# Nota: Usamos las mismas carpetas 'templates' y 'static'
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
GENERATED_ART_FOLDER = os.path.join(app.static_folder, 'generated_art')
os.makedirs(GENERATED_ART_FOLDER, exist_ok=True)

# ---------------------------------------------------
# CONFIGURACIÓN DEL DISPOSITIVO
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------
# CARGA DEL MODELO DE DIFUSIÓN (Generador de Imágenes)
# ---------------------------------------------------
print("Cargando modelo de difusión... (Esto puede tardar unos minutos)")

# Esta es la ruta donde tu 'train.py' guarda el modelo
#
RUTA_UNET_ENTRENADA = "models/diffusion_art_model" 

MODELO_BASE_ID = "CompVis/stable-diffusion-v1-4"

try:
    # Cargar el VAE (mejora la calidad)
    vae = AutoencoderKL.from_pretrained(MODELO_BASE_ID, subfolder="vae", torch_dtype=torch_dtype)

    # Cargar la U-Net (la 'CNN') que TÚ entrenaste
    unet = UNet2DConditionModel.from_pretrained(RUTA_UNET_ENTRENADA, torch_dtype=torch_dtype)

    # Cargar el pipeline completo con tu U-Net personalizada
    pipeline_difusion = DiffusionPipeline.from_pretrained(
        MODELO_BASE_ID,
        vae=vae,
        unet=unet,
        torch_dtype=torch_dtype,
        safety_checker=None # Desactiva el chequeo de seguridad
    )
    pipeline_difusion = pipeline_difusion.to(device)
    print(f"✅ Modelo de Difusión (UNet) cargado desde {RUTA_UNET_ENTRENADA}")

except Exception as e:
    print(f"❌ ERROR AL CARGAR EL MODELO DE DIFUSIÓN: {e}")
    print("Asegúrate de que la ruta 'models/diffusion_art_model' exista y contenga los archivos del modelo entrenado.")
    pipeline_difusion = None

# ---------------------------------------------------
# RUTA PRINCIPAL (/)
# ---------------------------------------------------
@app.route('/')
def home():
    # Renderiza la página del generador
    return render_template('generate.html')

# ---------------------------------------------------
# RUTA API PARA GENERAR IMÁGENES
# ---------------------------------------------------
@app.route('/generate-image', methods=['POST'])
def generate_image_api():
    if pipeline_difusion is None:
        return jsonify({"error": "El modelo de difusión no se cargó correctamente. Revisa la consola."}), 500

    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No se recibió ningún prompt"}), 400

    print(f"🖌️ [Generador] Generando imagen para: '{prompt}'")
    
    try:
        with torch.no_grad():
            imagen_generada = pipeline_difusion(prompt, num_inference_steps=50).images[0]
        
        # Guardar la imagen
        img_filename = f"art_gen_{hash(prompt)}.png" 
        img_save_path = os.path.join(GENERATED_ART_FOLDER, img_filename)
        imagen_generada.save(img_save_path)
        
        # Crear la URL
        image_url = url_for('static', filename=f'generated_art/{img_filename}')
        
        print(f"🖼️ [Generador] Imagen guardada en: {image_url}")
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        print(f"❌ ERROR en /generate-image: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == '__main__':
    # ¡IMPORTANTE! Lo ejecutamos en el puerto 5001 para no chocar con tu main.py original
    print("🚀 Servidor de PRUEBA DE IMÁGENES iniciado en http://127.0.0.1:5001")
    app.run(debug=True, threaded=False, port=5001)