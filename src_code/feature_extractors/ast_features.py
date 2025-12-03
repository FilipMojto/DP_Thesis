import ast
from radon.complexity import cc_visit

def count_ast_nodes(code: str) -> int:
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return 0


def extract_ast_features(diff_text: str) -> dict:
    added_lines = "\n".join(
        line[1:] for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    removed_lines = "\n".join(
        line[1:] for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )

    return {
        "ast_added_nodes": count_ast_nodes(added_lines),
        "ast_removed_nodes": count_ast_nodes(removed_lines),
    }


def complexity_from_code(code: str) -> int:
    try:
        blocks = cc_visit(code)
        return sum(b.complexity for b in blocks)
    except:
        return 0


def extract_complexity_features(diff_text: str) -> dict:
    added = "\n".join(
        l[1:] for l in diff_text.splitlines()
        if l.startswith("+") and not l.startswith("+++")
    )
    removed = "\n".join(
        l[1:] for l in diff_text.splitlines()
        if l.startswith("-") and not l.startswith("---")
    )

    return {
        "complexity_added": complexity_from_code(added),
        "complexity_removed": complexity_from_code(removed)
    }