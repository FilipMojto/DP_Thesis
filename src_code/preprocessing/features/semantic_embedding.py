
from git import Commit
import torch
import logging

from src_code.external_models.codebert import EMBED_MODEL

logging.basicConfig(level=logging.INFO)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# EMBED_MODEL = RobertaModel.from_pretrained("microsoft/codebert-base")
# EMBED_MODEL.to(device)



def calculate_semantic_embeddings(c: Commit, diff_text) -> dict:
    semantic_embedding = {"code_embed": None, "msg_embed": None}

    # --- Message and code embeddings ---
    if EMBED_MODEL:
        code_text = "\n".join(
            d.b_blob.data_stream.read().decode('utf-8', errors='ignore')[:2000]
            for d in diff_text if d.b_blob
        )
        msg_text = c.message
        # features["code_embed"] = EMBED_MODEL.encode([code_text])[0].tolist()
        # features["msg_embed"] = EMBED_MODEL.encode([msg_text])[0].tolist()
        # Code embedding
        inputs_code = tokenizer(code_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs_code = {k: v.to(device) for k, v in inputs_code.items()}
        with torch.no_grad():
            outputs_code = EMBED_MODEL(**inputs_code)
            code_embedding = outputs_code.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        # features["code_embed"] = code_embedding.tolist()
        semantic_embedding["code_embed"] = code_embedding.tolist()

        # Message embedding
        inputs_msg = tokenizer(msg_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs_msg = {k: v.to(device) for k, v in inputs_msg.items()}
        with torch.no_grad():
            outputs_msg = EMBED_MODEL(**inputs_msg)
            msg_embedding = outputs_msg.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        # features["msg_embed"] = msg_embedding.tolist()
        semantic_embedding["msg_embed"] = msg_embedding.tolist()
    else:
        logging.warning("Embedding model not available. Skipping semantic embedding features.")

    return semantic_embedding
    # else:
    #     features["code_embed"] = None
    #     features["msg_embed"] = None
