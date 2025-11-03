import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import hashlib
from flask import Flask, render_template, jsonify, url_for
import threading
import gradio as gr

# ---------------------------------------------------
# CONFIGURACIÓN FLASK
# ---------------------------------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ---------------------------------------------------
# CARPETA PARA GUARDAR IMÁGENES
# ---------------------------------------------------
GENERATED_ART_FOLDER = os.path.join(app.static_folder, 'generated_art')
os.makedirs(GENERATED_ART_FOLDER, exist_ok=True)

# ---------------------------------------------------
# CONFIGURACIÓN DEL DISPOSITIVO
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------
# CARGA DEL MODELO DE DIFUSIÓN
# ---------------------------------------------------
HF_REPO_ID = "Joseph1112/DibujoArte"
BASE_MODEL = "CompVis/stable-diffusion-v1-4"

print("Cargando modelo de difusión...")
try:
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(HF_REPO_ID, torch_dtype=torch_dtype)
    pipeline_difusion = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        vae=vae,
        unet=unet,
        torch_dtype=torch_dtype,
        safety_checker=None
    ).to(device)
    print(f"✅ Modelo cargado: {HF_REPO_ID}")
except Exception as e:
    print(f"❌ ERROR al cargar modelo: {e}")
    pipeline_difusion = None

# ---------------------------------------------------
# FUNCIÓN DE GENERACIÓN DE IMÁGENES
# ---------------------------------------------------
def generar_imagen(prompt):
    if pipeline_difusion is None or not prompt:
        return None
    
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    img_filename = f"arte_{prompt_hash[:8]}.png"
    img_save_path = os.path.join(GENERATED_ART_FOLDER, img_filename)

    if not os.path.exists(img_save_path):
        with torch.no_grad():
            imagen = pipeline_difusion(prompt, num_inference_steps=50).images[0]
        imagen.save(img_save_path)
    
    return img_save_path

# ---------------------------------------------------
# RUTA PRINCIPAL (HTML)
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template("chat.html")

# ---------------------------------------------------
# ENDPOINT PARA JS
# ---------------------------------------------------
@app.route('/api/generate', methods=['POST'])
def api_generate():
    from flask import request
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt received"}), 400
    try:
        img_path = generar_imagen(prompt)
        img_url = url_for('static', filename=f'generated_art/{os.path.basename(img_path)}')
        return jsonify({"image_url": img_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------
# FUNCIONES GRADIO (Solo backend opcional)
# ---------------------------------------------------
def gradio_generate(prompt):
    return generar_imagen(prompt)

def launch_gradio():
    with gr.Blocks() as gr_app:
        gr.Interface(
            fn=gradio_generate,
            inputs=gr.Textbox(placeholder="Escribe tu prompt aquí...", label="Prompt"),
            outputs=gr.Image(type="filepath"),
            live=False
        ).launch(server_name="0.0.0.0", server_port=7861, share=True)

# Lanzar Gradio en un hilo separado
threading.Thread(target=launch_gradio, daemon=True).start()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    # Flask sirve tu frontend HTML/JS/CSS
    app.run(host='0.0.0.0', port=5000)
