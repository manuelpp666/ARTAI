import torch
import torch.nn.functional as F
from transformer import Transformer
from preprocess import codificar, decodificar

def generar_texto(modelo: Transformer, texto_inicio: str, longitud=200, temperatura=1.2, seq_len=50, device='cpu', stoi=None, itos=None):
    """Genera texto autoregresivo usando el transformer."""
    modelo.eval()
    generado = codificar(texto_inicio, stoi)
    for _ in range(longitud):
        x = torch.tensor([generado[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = modelo(x)
            logits = logits[0, -1, :] / temperatura
            probs = F.softmax(logits, dim=-1)
            token_siguiente = torch.multinomial(probs, 1).item()
        generado.append(token_siguiente)
    return decodificar(generado, itos)
