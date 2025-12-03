import re

FIX_WORDS = ["fix", "bug", "issue", "solve"]

def extract_msg_features(message: str) -> dict:
    lower = message.lower()

    return {
        "msg_len": len(message),
        "has_fix_kw": any(w in lower for w in FIX_WORDS),
        "num_capital_words": sum(1 for w in message.split() if w.isupper()),
    }