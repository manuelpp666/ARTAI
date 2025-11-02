import torch
import torch.nn.functional as F

# ============================================================
# 🎯 Muestreo controlado con top-k y top-p (nucleus sampling)
# ============================================================
def sample_next_token(logits, top_k=50, top_p=0.9, temperature=0.7):
    probs = F.softmax(logits / temperature, dim=-1)

    # 🔹 top-k
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)

    # Renormalizar
    probs = probs / probs.sum()

    # 🔹 top-p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative_probs > top_p] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()

    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token)


# ============================================================
# 🧠 Generación de texto autoregresiva factual
# ============================================================
def generar_texto(modelo, tokenizer, device, seed_text, max_length=200,
                  top_k=40, top_p=0.9, temperature=0.6, repetition_penalty=1.15):
    """
    Genera texto factual tipo Wikipedia.
    Se detiene automáticamente si aparece [FIN_SECCION].
    """
    modelo.eval()
    tokens = tokenizer.encode(seed_text).ids
    generados = tokens.copy()
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    fin_token_id = tokenizer.token_to_id("[FIN_SECCION]")

    with torch.no_grad():
        for _ in range(max_length):
            logits = modelo(input_ids)[:, -1, :]

            # 🔹 Penalización de repetición leve
            for token_id in set(generados):
                if token_id < logits.size(-1):
                    logits[0, token_id] /= repetition_penalty

            # 🔹 Muestreo controlado
            next_token = sample_next_token(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            next_token = next_token.to(device)
            token_id = next_token.item()

            # 🔹 Detener si aparece token de fin de sección
            if fin_token_id is not None and token_id == fin_token_id:
                break

            generados.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generados)
