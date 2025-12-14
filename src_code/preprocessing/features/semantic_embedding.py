from git import Commit
import torch
import logging

from src_code.external_models.codebert import EMBED_MODEL, tokenizer, device

logging.basicConfig(level=logging.INFO)


def calculate_semantic_embeddings(c: Commit, diff_text) -> dict:
    semantic_embedding = {"code_embed": None, "msg_embed": None}

    # --- Message and code embeddings ---
    if EMBED_MODEL:
        code_text = "\n".join(
            d.b_blob.data_stream.read().decode("utf-8", errors="ignore")[:2000]
            for d in diff_text
            if d.b_blob
        )
        msg_text = c.message
        # features["code_embed"] = EMBED_MODEL.encode([code_text])[0].tolist()
        # features["msg_embed"] = EMBED_MODEL.encode([msg_text])[0].tolist()
        # Code embedding
        # The text (code or message) is first broken down into numerical tokens that the model understands.
        # Padding & Truncation: Since Transformer models have a fixed input size (max_length=512), longer inputs are truncated,
        # and shorter inputs are padded to match the length.
        inputs_code = tokenizer(
            code_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs_code = {k: v.to(device) for k, v in inputs_code.items()}
        # forward pass: The token IDs are passed through the pre-trained CodeBERT model.
        with torch.no_grad():
            # This context manager is used because you are performing inference (just calculating features), not training.
            # This saves memory and speeds up computation.

            # The outputs_code object contains the final hidden states, which is a tensor representing a 768-dimensional vector
            # for every single input token.
            outputs_code = EMBED_MODEL(**inputs_code)
            # Since the output contains an embedding for every token, you need to aggregate these into a single fixed-size vector
            # for the entire input.
            code_embedding = outputs_code.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        # features["code_embed"] = code_embedding.tolist()
        semantic_embedding["code_embed"] = code_embedding.tolist()

        # Message embedding
        inputs_msg = tokenizer(
            msg_text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs_msg = {k: v.to(device) for k, v in inputs_msg.items()}
        with torch.no_grad():
            outputs_msg = EMBED_MODEL(**inputs_msg)
            msg_embedding = outputs_msg.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        # features["msg_embed"] = msg_embedding.tolist()
        semantic_embedding["msg_embed"] = msg_embedding.tolist()
    else:
        logging.warning(
            "Embedding model not available. Skipping semantic embedding features."
        )

    return semantic_embedding
    # else:
    #     features["code_embed"] = None
    #     features["msg_embed"] = None
