from src_code.external_models.codebert import EMBED_MODEL, tokenizer, device
import torch


def mean_pool(hidden_states, attention_mask):
    """Correct mean pooling that ignores padding."""
    masked = hidden_states * attention_mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1).unsqueeze(-1)
    return summed / counts


def compute_context_embedding(diff_text):
    """Compute mean-pooled embedding of surrounding code context for added lines."""
    if not EMBED_MODEL:
        return None

    contexts = []
    for d in diff_text:
        try:
            patch = d.diff.decode(errors="ignore")
        except Exception:
            continue

        lines = patch.splitlines()
        for i, line in enumerate(lines):
            if line.startswith('+') and not line.startswith('+++'):
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                snippet = '\n'.join(lines[start:end])
                # contexts.append(snippet[:512])  # truncate context window
                contexts.append(snippet)

    if not contexts:
        return None

    # Embed and mean-pool
    # embeddings = EMBED_MODEL.encode(contexts, show_progress_bar=False)
    inputs = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # outputs = EMBED_MODEL(**inputs)
        # embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        outputs = EMBED_MODEL(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = pooled.cpu().numpy()
        
    # return np.mean(embeddings, axis=0).tolist()

    # return one embedding per commit
    return pooled.mean(axis=0).tolist()


### ---------- TOKEN FEATURES ----------
def count_token_keywords(diff_text):
    """Count keywords like TODO, FIXME, try, except, raise in added lines."""
    tokens = {
        "todo": 0,
        "fixme": 0,
        "try": 0,
        "except": 0,
        "raise": 0,
    }
    for d in diff_text:
        patch = d.diff.decode(errors="ignore")
        for line in patch.splitlines():
            if line.startswith('+') and not line.startswith('+++'):  # only added lines
                l = line.lower()
                tokens["todo"] += l.count("todo")
                tokens["fixme"] += l.count("fixme")
                tokens["try"] += l.count("try")
                tokens["except"] += l.count("except")
                tokens["raise"] += l.count("raise")
    return tokens