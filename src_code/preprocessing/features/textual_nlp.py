def compute_msg_flags(msg):
    msg_lower = msg.lower()
    return {
        "msg_len": len(msg),
        "has_fix_kw": int("fix" in msg_lower),
        "has_bug_kw": int("bug" in msg_lower)
    }