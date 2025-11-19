# ================================================================
# main.py — Interfaz Flask para ArtAI (Optimizado)
# ================================================================
from flask import Flask, render_template, request, jsonify
from gradio_client import Client
import os
from PIL import Image

# ---------------------------
# Modelos internos
# ---------------------------
from models_utils.arte_desde_cero import cargar_modelo_desde_cero, generar_texto
from models_utils.arte_loader import get_qa

# ---------------------------
# Inicialización Flask
# ---------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ================================================================
# 1️⃣ CLIENTE HUGGING FACE (IMAGEN)
# ================================================================
HF_REPO_ID = "Joseph1112/ArtAI"
try:
    client = Client(HF_REPO_ID)
    print(f"✅ Cliente Hugging Face inicializado con: {HF_REPO_ID}")
except Exception as e:
    print(f"⚠️ Error conectando a HuggingFace: {e}")
    client = None

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
# 3️⃣ CARGAR MODELO PHI-3 + FAISS
# ================================================================
print("🧠 Cargando QA preentrenado + FAISS (modo cache RAM)...")
try:
    qa = get_qa()
    print("✅ Modelo Phi-3 + FAISS listo.")
except Exception as e:
    qa = None
    print(f"❌ Error crítico cargando QA: {e}")


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
        if not client:
             return jsonify({"error": "El servicio de imágenes no está disponible."}), 503
             
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
                # Asumimos que es un objeto PIL
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
        if qa is None:
            return jsonify({"tipo": "texto", "texto": "⚠️ El modelo preentrenado no pudo cargarse."})

        try:
            print(f"🧠 Consultando modelo preentrenado: {mensaje}")
            
            # Invocamos al modelo
            result = qa.invoke({"query": mensaje})
            
            # Extraemos solo el resultado
            texto_final = result["result"]
            
            # Limpieza extra de seguridad: Si el modelo repite etiquetas, las quitamos
            if "<|assistant|>" in texto_final:
                texto_final = texto_final.split("<|assistant|>")[-1]
            
            # Quitamos espacios vacíos al inicio/final
            texto_final = texto_final.strip()

            return jsonify({"tipo": "texto", "texto": texto_final})

        except Exception as e:
            print(f"ERROR en texto_pre: {e}")
            return jsonify({"error": f"❌ Error al generar texto: {str(e)}"}), 500

    else:
        return jsonify({"error": f"Modo desconocido: {modo}"}), 400


# ================================================================
# 5️⃣ EJECUTAR APP
# ================================================================
if __name__ == '__main__':
    app.run(debug=False)