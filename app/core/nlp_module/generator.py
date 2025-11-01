import torch
import torch.nn.functional as F

def generar_texto(modelo, tokenizer, device, seed_text, max_length=200,
                  top_p=0.9, repetition_penalty=1.2, temperature=1.0):
    modelo.eval()
    tokens = tokenizer.encode(seed_text).ids
    generados = tokens.copy()
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            logits = modelo(input_ids)[:, -1, :] / temperature

            # Penalidad de repetición
            for token_id in set(generados):
                if token_id < logits.size(-1):
                    logits[0, token_id] /= repetition_penalty

            # Nucleus sampling (top-p)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumprobs > top_p
            sorted_probs[cutoff] = 0
            if sorted_probs.sum() > 0:
                sorted_probs /= sorted_probs.sum()
            else:
                sorted_probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, next_token)

            generados.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generados)
