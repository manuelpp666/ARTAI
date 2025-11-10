# ================================================================
# main.py — Interfaz Flask para ArtAI (solo generador de imágenes)
# ================================================================
from flask import Flask, render_template, request, jsonify
from gradio_client import Client

# ---------------------------------------------------
# CONFIGURACIÓN FLASK
# ---------------------------------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ---------------------------------------------------
# CONFIGURACIÓN CLIENTE HUGGING FACE
# ---------------------------------------------------
HF_REPO_ID = "Joseph1112/ArtAI"  # Tu Space o modelo en Hugging Face
client = Client(HF_REPO_ID)

print(f"✅ Cliente Hugging Face inicializado con: {HF_REPO_ID}")

# ---------------------------------------------------
# RUTAS FLASK
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    modo = data.get("modo")  # imagen / texto_cero / texto_pre
    mensaje = data.get("message", "")

    # ---------------------------------------
    # 🖼️ GENERADOR DE IMÁGENES DE ARTE
    # ---------------------------------------
    if modo == "imagen":
        try:
            print(f"🎨 Generando imagen para prompt: {mensaje}")
            result = client.predict(prompt=mensaje, api_name="/predict")

            # Si es una URL, la devolvemos directo
            if isinstance(result, str) and result.startswith("http"):
                return jsonify({"tipo": "imagen", "url": result})

            # Si no es URL, puede ser ruta o PIL.Image
            from PIL import Image
            import os

            img_folder = os.path.join(app.static_folder, "generated")
            os.makedirs(img_folder, exist_ok=True)

            img_path = os.path.join(img_folder, f"arte_{hash(mensaje)}.png")

            if isinstance(result, str) and os.path.exists(result):
                # Si es una ruta de archivo local
                Image.open(result).save(img_path)
            else:
                # Si es un objeto tipo PIL.Image o similar
                try:
                    result.save(img_path)
                except Exception:
                    return jsonify({"error": "No se pudo procesar la imagen generada"}), 500

            # Devuelve una URL accesible desde el navegador
            img_url = f"/static/generated/{os.path.basename(img_path)}"
            return jsonify({"tipo": "imagen", "url": img_url})

        except Exception as e:
            return jsonify({"error": f"Error al generar imagen: {str(e)}"}), 500
    # ---------------------------------------
    # ✍️ MODO TEXTO (Placeholder)
    # ---------------------------------------
    elif modo in ("texto_cero", "texto_pre"):
        return jsonify({
            "tipo": "texto",
            "texto": f"⚠️ Modo '{modo}' deshabilitado temporalmente. Solo disponible el generador de imágenes."
        })

    else:
        return jsonify({"error": f"Modo desconocido: {modo}"}), 400


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
