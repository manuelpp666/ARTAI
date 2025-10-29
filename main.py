from flask import Flask, render_template, request, jsonify
import torch
from app.core.nlp_module.transformer import Transformer
from app.core.nlp_module.generator import generar_texto
from app.core.nlp_module.preprocess import cargar_vocab

# ---------------------------------------------------
# CONFIGURACIÓN FLASK
# ---------------------------------------------------
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# ---------------------------------------------------
# CONFIGURACIÓN DEL DISPOSITIVO
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# CARGA DEL VOCABULARIO
# ---------------------------------------------------
ruta_vocab = "models/vocab_art.pt"  # cambia si lo guardaste en otra ruta
stoi, itos = cargar_vocab(ruta_vocab)
vocab_size = len(stoi)
seq_len = 50

# ---------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------
modelo = Transformer(vocab_size=vocab_size).to(device)

ruta_modelo = "models/transformer_art_model.pth"
checkpoint = torch.load(ruta_modelo, map_location=device)
modelo.load_state_dict(checkpoint["modelo"])

modelo.eval()
print(f"✅ Modelo cargado correctamente desde {ruta_modelo}")
print(f"✅ Vocabulario con {vocab_size} caracteres cargado.")

# ---------------------------------------------------
# RUTAS FLASK
# ---------------------------------------------------
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")

    # Generar texto coherente
    respuesta_texto = generar_texto(
        modelo=modelo,
        texto_inicio=user_message,
        longitud=200,        # puedes ajustarlo (100-300 suele ir bien)
        temperatura=0.9,     # menor → más coherente, mayor → más creativo
        seq_len=seq_len,
        device=device,
        stoi=stoi,
        itos=itos
    )

    return jsonify({"text": respuesta_texto})

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
