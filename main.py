# ================================================================
# main.py — Interfaz Flask para ArtAI (imagen + texto_cero LSTM)
# ================================================================
from flask import Flask, render_template, request, jsonify
from gradio_client import Client
import os
from models_utils.arte_desde_cero import cargar_modelo_desde_cero, generar_texto

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ---------------------------------------------------
# CLIENTE HUGGING FACE (modo imagen)
# ---------------------------------------------------
HF_REPO_ID = "Joseph1112/ArtAI"
client = Client(HF_REPO_ID)
print(f"✅ Cliente Hugging Face inicializado con: {HF_REPO_ID}")

# ---------------------------------------------------
# CARGAR MODELO DE TEXTO DESDE CERO (LSTM)
# ---------------------------------------------------
MODEL_PATH = "models/arte/entrenamiento_desde_cero/v_cero.pth"

try:
    modelo_cero, word2idx, idx2word = cargar_modelo_desde_cero(MODEL_PATH)
    print("✅ Modelo de texto (LSTM desde cero) cargado correctamente.")
except Exception as e:
    modelo_cero, word2idx, idx2word = None, None, None
    print(f"⚠️ No se pudo cargar el modelo desde cero: {e}")

# ---------------------------------------------------
# RUTAS FLASK
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    modo = data.get("modo")
    mensaje = data.get("message", "")

    # =================== IMAGEN ===================
    if modo == "imagen":
        try:
            print(f"🎨 Generando imagen para prompt: {mensaje}")
            result = client.predict(prompt=mensaje, api_name="/predict")

            from PIL import Image
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

    # =================== TEXTO DESDE CERO ===================
    elif modo == "texto_cero":
        if modelo_cero is None:
            return jsonify({"tipo": "texto", "texto": "⚠️ El modelo desde cero no está cargado."})

        try:
            respuesta = generar_texto(modelo_cero, word2idx, idx2word, mensaje, max_len=60)
            return jsonify({"tipo": "texto", "texto": respuesta})
        except Exception as e:
            return jsonify({"error": f"❌ Error al generar texto: {str(e)}"}), 500

    # =================== TEXTO PREENTRENADO (a futuro) ===================
    elif modo == "texto_pre":
        return jsonify({
            "tipo": "texto",
            "texto": "⚠️ Modo 'texto_pre' aún no implementado."
        })

    else:
        return jsonify({"error": f"Modo desconocido: {modo}"}), 400


if __name__ == '__main__':
    app.run(debug=True)
