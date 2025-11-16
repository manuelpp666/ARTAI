# ================================================================
# main.py — Interfaz Flask para ArtAI (imagen + texto_cero + texto_pre)
# ================================================================
from flask import Flask, render_template, request, jsonify
from gradio_client import Client
import os
from PIL import Image

# ---------------------------
# Modelos internos
# ---------------------------
from models_utils.arte_desde_cero import cargar_modelo_desde_cero, generar_texto
from models_utils.arte_loader import get_qa   # <-- OPTIMIZADO (modelo cargado 1 sola vez)

# ---------------------------
# Inicialización Flask
# ---------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ================================================================
# 1️⃣ CLIENTE HUGGING FACE (IMAGEN)
# ================================================================
HF_REPO_ID = "Joseph1112/ArtAI"
client = Client(HF_REPO_ID)
print(f"✅ Cliente Hugging Face inicializado con: {HF_REPO_ID}")

# ================================================================
# 2️⃣ CARGAR MODELO LSTM DESDE CERO
# ================================================================
MODEL_PATH = "models/arte/entrenamiento_desde_cero/v_cero.pth"

try:
    modelo_cero, word2idx, idx2word = cargar_modelo_desde_cero(MODEL_PATH)
    print("✅ Modelo de texto (LSTM desde cero) cargado correctamente.")
except Exception as e:
    modelo_cero, word2idx, idx2word = None, None, None
    print(f"⚠️ No se pudo cargar el modelo desde cero: {e}")

# ================================================================
# 3️⃣ CARGAR MODELO PHI-3 + FAISS SOLO UNA VEZ (CACHEADO)
# ================================================================
print("🧠 Cargando QA preentrenado + FAISS (modo cache RAM)...")
qa = get_qa()   # <-- Aquí ya está todo cargado UNA SOLA VEZ
print("✅ Modelo Phi-3 + FAISS listo.")


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
    # MODO IMAGEN
    # ------------------------------------------------------------
    if modo == "imagen":
        try:
            print(f"🎨 Generando imagen para prompt: {mensaje}")
            result = client.predict(prompt=mensaje, api_name="/predict")

            img_folder = os.path.join(app.static_folder, "generated")
            os.makedirs(img_folder, exist_ok=True)
            img_path = os.path.join(img_folder, f"arte_{hash(mensaje)}.png")

            if isinstance(result, str) and result.startswith("http"):
                return jsonify({"tipo": "imagen", "url": result})

            elif isinstance(result, str) and os.path.exists(result):
                Image.open(result).save(img_path)

            else:
                result.save(img_path)

            img_url = f"/static/generated/{os.path.basename(img_path)}"
            return jsonify({"tipo": "imagen", "url": img_url})

        except Exception as e:
            return jsonify({"error": f"Error al generar imagen: {str(e)}"}), 500

    # ------------------------------------------------------------
    # MODO TEXTO LSTM DESDE CERO
    # ------------------------------------------------------------
    elif modo == "texto_cero":
        if modelo_cero is None:
            return jsonify({"tipo": "texto", "texto": "⚠️ El modelo desde cero no está cargado."})

        try:
            respuesta = generar_texto(modelo_cero, word2idx, idx2word, mensaje, max_len=60)
            return jsonify({"tipo": "texto", "texto": respuesta})

        except Exception as e:
            return jsonify({"error": f"❌ Error al generar texto: {str(e)}"}), 500

    # ------------------------------------------------------------
    # MODO TEXTO PREENTRENADO (PHI-3 + FAISS)
    # ------------------------------------------------------------
    elif modo == "texto_pre":
        try:
            print(f"🧠 Consultando modelo preentrenado: {mensaje}")
            result = qa.invoke({"query": mensaje})
            respuesta = result["result"]

            return jsonify({"tipo": "texto", "texto": respuesta})

        except Exception as e:
            return jsonify({"error": f"❌ Error al generar texto: {str(e)}"}), 500

    else:
        return jsonify({"error": f"Modo desconocido: {modo}"}), 400


# ================================================================
# 5️⃣ EJECUTAR APP (sin doble carga)
# ================================================================
if __name__ == '__main__':
    app.run(debug=False)   # <-- IMPORTANTE: evita doble carga del modelo
