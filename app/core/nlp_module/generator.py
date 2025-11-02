import torch
import torch.nn.functional as F

# ============================================================
# Muestreo controlado con top-k y top-p (nucleus sampling)
# ============================================================
def sample_next_token(logits, top_k=50, top_p=0.9, temperature=0.8):
    # Aplicar temperatura
    probs = F.softmax(logits / temperature, dim=-1)

    # 🔹 top-k: mantener solo los k tokens más probables
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)
    
    # Renormalización segura
    sum_probs = probs.sum()
    if sum_probs <= 0 or torch.isnan(sum_probs) or torch.isinf(sum_probs):
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / sum_probs

    # 🔹 top-p (nucleus): mantener tokens acumulando probabilidad ≤ top_p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative_probs > top_p] = 0

    # Renormalización segura
    sum_probs = sorted_probs.sum()
    if sum_probs <= 0 or torch.isnan(sum_probs) or torch.isinf(sum_probs):
        sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.numel()
    else:
        sorted_probs = sorted_probs / sum_probs

    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token)


# ============================================================
# 🧠 Generación de texto autoregresiva
# ============================================================
def generar_texto(modelo, tokenizer, device, seed_text, max_length=200,
                  top_k=50, top_p=0.9, temperature=0.8, repetition_penalty=1.25):
    modelo.eval()
    tokens = tokenizer.encode(seed_text).ids
    generados = tokens.copy()
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            logits = modelo(input_ids)[:, -1, :]

            # 🔹 Penalización de repetición (reduce tokens ya usados)
            for token_id in set(generados):
                if token_id < logits.size(-1):
                    logits[0, token_id] /= repetition_penalty
                    logits[0, token_id] = max(logits[0, token_id], -1e9)  # evita demasiado negativo

            # 🔹 Obtener siguiente token con muestreo controlado
            next_token = sample_next_token(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            next_token = next_token.to(device)

            # 🔹 Actualizar secuencia
            generados.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generados)
