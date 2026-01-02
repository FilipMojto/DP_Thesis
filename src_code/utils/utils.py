import re

def is_embedding_column(col_name: str) -> bool:
    """
    Returns True if column name ends with emb_<positive_integer>
    Examples:
        emb_0        -> True
        code_emb_12 -> True
        msg_emb_768 -> True
        emb_        -> False
        emb_x       -> False
    """
    return bool(re.search(r"emb_\d+$", col_name))