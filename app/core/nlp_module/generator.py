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
        probs = probs / probs.sum()  # renormalizar

    # 🔹 top-p (nucleus): mantener tokens acumulando probabilidad ≤ top_p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()  # renormalizar
    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token)


# ============================================================
# 🧠 Generación de texto autoregresiva
# ============================================================
def generar_texto(modelo, tokenizer, device, seed_text, max_length=200,
                  top_k=50, top_p=0.9, temperature=0.8, repetition_penalty=1.2):
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

            # 🔹 Obtener siguiente token con muestreo controlado
            next_token = sample_next_token(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            next_token = next_token.to(device)

            # 🔹 Actualizar secuencia
            generados.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generados)
