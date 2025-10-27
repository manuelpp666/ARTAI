from flask import Flask, render_template, request, jsonify
import time
from app.core.nlp_module.transformer import Transformer
from app.core.nlp_module.generator import generar_texto
import torch

app = Flask(__name__, template_folder='app/templates',static_folder='app/static')

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros según tu entrenamiento
vocab_size = len("abcdefghijklmnopqrstuvwxyzáéíóúü ,.!?\n")  # igual que tu preprocess
seq_len = 50

# Inicializar y cargar modelo entrenado
modelo = Transformer(vocab_size=vocab_size).to(device)
checkpoint = torch.load("models/transformer_art_model.pth", map_location=device)
modelo.load_state_dict(checkpoint["modelo"])

modelo.eval()
print("Modelo Transformer cargado y listo.")

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")

    # Generar texto usando tu Transformer
    respuesta_texto = generar_texto(
        modelo=modelo,
        texto_inicio=user_message,
        longitud=150,          # longitud del texto a generar
        temperatura=1.0,
        seq_len=seq_len,
        device=device
    )

    response = {
        "text": respuesta_texto
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
