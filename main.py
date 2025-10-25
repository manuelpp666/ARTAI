from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__, template_folder='app/templates',static_folder='app/static')

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")

    # Aquí luego conectarás tu modelo Transformer + Diffusion.
    # Por ahora simulamos respuesta:
    if "pintura" in user_message.lower():
        response = {
            "text": "Aquí tienes una pintura inspirada en tu descripción.",
            "image_url": "/static/images/sample_art.jpg"
        }
    else:
        response = {
            "text": f"Reflexionemos sobre eso... el arte también es {user_message.lower()}."
        }

    time.sleep(1)  # Pequeña pausa para efecto “pensando”
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
