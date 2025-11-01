import torch
import torch.nn.functional as F
from transformer import Transformer
from preprocess import codificar, decodificar

def generar_texto(
    modelo: Transformer,
    texto_inicio: str,
    longitud: int = 200,
    temperatura: float = 1.2,
    seq_len: int = 50,
    device: str = 'cpu',
    tokenizer=None,
    top_k: int = 50,
    top_p: float = None
):
    """
    Genera texto autoregresivo usando el transformer.
    Compatible con tokenizador BPE.
    """
    modelo.eval()
    generado = codificar(texto_inicio, tokenizer)

    for _ in range(longitud):
        x = torch.tensor([generado[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = modelo(x)
            logits = logits[0, -1, :]  # Solo último token
            token_siguiente = sample_logits(logits, temperatura, top_k, top_p)
        generado.append(token_siguiente)

    return decodificar(generado, tokenizer)


def sample_logits(logits, temperatura=1.0, top_k: int = 50, top_p: float = None):
    """
    Muestra un token a partir de los logits, usando top-k o top-p sampling.
    """
    logits = logits / temperatura

    if top_k is not None:
        # Top-K sampling
        valores, indices = torch.topk(logits, k=top_k)
        probs = F.softmax(valores, dim=-1)
        idx = torch.multinomial(probs, 1)
        return indices.gather(-1, idx).item()

    elif top_p is not None:
        # Nucleus / Top-P sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_logits[cumulative_probs > top_p] = -float("Inf")
        probs = F.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        return sorted_indices.gather(-1, idx).item()

    else:
        # Muestreo simple softmax
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
