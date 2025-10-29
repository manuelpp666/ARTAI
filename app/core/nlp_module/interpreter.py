# ---------------------------------------------------
# interpreter.py — Detector semántico de intención (solo PyTorch)
# ---------------------------------------------------
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------
# Cargar modelo semántico liviano (solo PyTorch)
# ---------------------------------------------------
# all-MiniLM-L6-v2 es rápido, preciso y no requiere TensorFlow
modelo_semantico = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---------------------------------------------------
# Catálogo de intenciones base
# ---------------------------------------------------
INTENCIONES = {
    "imagen": [
        "dibuja un cuadro",
        "genera una ilustración",
        "píntame una obra",
        "crea una imagen de arte",
        "representa visualmente",
        "quiero una pintura",
        "haz un retrato",
        "muéstrame una obra de arte",
        "visualiza esta escena",
        "genera arte visual",
        "dibuja a un pintor",
        "crea una escultura digital",
        "realiza una composición visual",
        "haz un boceto",
        "píntame algo",
    ],
    "texto": [
        "qué es el arte",
        "descríbeme la pintura de",
        "explícame el estilo de",
        "cuál es la historia del arte",
        "cuéntame sobre un pintor",
        "resúmeme la obra de",
        "analiza la pintura",
        "interpreta la obra",
        "defíneme el arte",
        "dime qué significa",
        "dime la biografía de",
        "enséñame sobre arte",
        "qué representa esta pintura",
        "háblame de la corriente artística",
        "explica el movimiento artístico",
        "quién fue este pintor",
    ],
    "otro": [
        "hola",
        "gracias",
        "adiós",
        "quién eres",
        "cómo estás",
        "buenos días",
        "qué puedes hacer",
        "ayuda",
    ]
}

# ---------------------------------------------------
# Precalcular embeddings base y normalizarlos
# ---------------------------------------------------
with torch.no_grad():
    embeddings_intenciones = {
        categoria: F.normalize(modelo_semantico.encode(frases, convert_to_tensor=True), p=2, dim=1)
        for categoria, frases in INTENCIONES.items()
    }

# ---------------------------------------------------
# Reglas heurísticas adicionales (sinónimos directos)
# ---------------------------------------------------
PALABRAS_IMAGEN = {
    "dibuja", "píntame", "ilustración", "retrátame", "visualiza", "crea una imagen",
    "haz una obra", "genera arte", "muéstrame una pintura"
}
PALABRAS_TEXTO = {
    "explica", "descríbeme", "cuéntame", "enséñame", "analiza", "resúmeme", "interpreta", "defíneme"
}

# ---------------------------------------------------
# Función principal mejorada
# ---------------------------------------------------
def detectar_intencion(texto_usuario: str, umbral=0.45):
    """
    Detecta si el usuario quiere generar una imagen, recibir texto o algo distinto.
    Devuelve: "imagen", "texto" o "otro"
    """
    texto = texto_usuario.lower().strip()

    # --- Heurística rápida mejorada
    contiene_texto = any(p in texto for p in PALABRAS_TEXTO)
    contiene_imagen = any(p in texto for p in PALABRAS_IMAGEN)

    if contiene_texto:
        return "texto"
    elif contiene_imagen:
        return "imagen"

    # --- Comparación semántica usando embeddings
    with torch.no_grad():
        embedding_usuario = F.normalize(
            modelo_semantico.encode(texto, convert_to_tensor=True), p=2, dim=0
        )

        puntajes = {}
        for categoria, emb_base in embeddings_intenciones.items():
            similitud = util.cos_sim(embedding_usuario, emb_base).max().item()
            puntajes[categoria] = similitud

        categoria_predicha = max(puntajes, key=puntajes.get)
        confianza = puntajes[categoria_predicha]

    # --- Umbral flexible
    if confianza < umbral:
        return "otro"

    # --- Corrección contextual adicional
    if categoria_predicha == "imagen" and contiene_texto:
        return "texto"

    return categoria_predicha


