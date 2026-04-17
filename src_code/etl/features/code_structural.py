import ast
import warnings
from radon.complexity import cc_visit
import re

# def count_ast_nodes(code):
#     """Counts how many nodes appear in the Python Abstract Syntax Tree (AST) for a given code snippet.

#     Args:
#         code (_type_): given code snippet

#     Returns:
#         _type_: the number of nodes
#     """
#     try:
#         # tree = ast.parse(code)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", SyntaxWarning)
#             tree = ast.parse(code)
#         return len(list(ast.walk(tree)))
#     except Exception:
#         return 0


# def get_complexity(code):
#     """Computes the cyclomatic complexity of the code using radon.

#     Cyclomatic complexity measures the number of independent execution paths (e.g. if, for, while, try, etc.).

#     Args:
#         code (_type_): code_snippet

#     Returns:
#         _type_: sums of individual complexities
#     """
#     try:
#         # cc_visit(code) finds all complexity blocks (functions, methods, classes)
#         blocks = cc_visit(code)
#         return sum(b.complexity for b in blocks)
#     except Exception:
#         return 0

# def get_functions_in_diff_range(code: str, changed_lines: set) -> list[str]:
#     """Identifies and extracts the full source code of functions that intersect
#     with the set of lines changed in the diff.

#     “Which functions were touched by this commit, and give me their code.”

#     Args:
#         code (str): _description_
#         changed_lines (set): _description_

#     Returns:
#         list[str]: a list of function source strings
#     """
#     functions_to_analyze = []
    
#     # 1. Parse the AST to find function/class definitions and their line numbers
#     try:
#         tree = ast.parse(code)
#     except Exception:
#         return []

#     # 2. Iterate over all nodes to find function/method definitions
#     for node in ast.walk(tree):
#         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#             # The function definition starts at node.lineno (1-based)
#             # Find the end line number (requires using the source code or a helper)
#             # For simplicity, we assume end is the last line of the function block.
#             # A perfect solution would use astor or similar tools, but we approximate:
            
#             start_line = node.lineno
#             end_line = getattr(node, 'end_lineno', start_line + 1) # Fallback if no end_lineno
            
#             # Check if the function body (including its signature) overlaps with changed lines
#             if any(line in changed_lines for line in range(start_line, end_line + 1)):
                
#                 # Extract the source code of the function body
#                 lines = code.splitlines()
#                 # ast lines are 1-based, list indices are 0-based
#                 func_lines = lines[start_line - 1 : end_line] 
#                 functions_to_analyze.append('\n'.join(func_lines))
    
#     return functions_to_analyze

# def extract_code_structural_features(diff_text):
#     """It computes structural change metrics from a Git diff.

#     Args:
#         diff_text (_type_): iterable of diff objects (likely from GitPython)

#     Returns:
#         _type_: _description_
#     """
#     ast_delta = 0
#     complexity_delta = 0
#     max_func_change = 0

#     for d in diff_text:
#         if not d.b_path or not d.b_blob:
#             continue

#         # 💡 Optimization Step 1: Skip non-Python files (optional but highly recommended)
#         if not d.b_path.endswith('.py'):
#             continue

#         try:
#             new_code = d.b_blob.data_stream.read().decode('utf-8', errors='ignore')
#             old_code = d.a_blob.data_stream.read().decode('utf-8', errors='ignore') if d.a_blob else ""
#         except Exception:
#             continue

#         changed_lines_new = set()

#         patch_text = d.diff.decode(errors="ignore")
        
#         # This is a basic way to get lines. A more robust way uses the diff library.
#         # It relies on reading the hunk headers (@@ -old_start,old_count +new_start,new_count @@)
        
#         for hunk_match in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch_text, re.MULTILINE):
#             new_start = int(hunk_match.group(1))
#             new_count = int(hunk_match.group(2) or 1)
#             changed_lines_new.update(range(new_start, new_start + new_count))


#         # 💡 Optimization Step 3: Analyze only the code of affected functions
        
#         # Analyze NEW code
#         new_functions = get_functions_in_diff_range(new_code, changed_lines_new)
        
#         # Analyze OLD code (Need to find the corresponding old lines, which is complex. 
#         # For simplicity in this fix, we analyze the *new* versions and the *old* versions
#         # corresponding to the *newly identified* functions.)
#         old_functions = []
#         if old_code:
#             # Re-run changed lines identification on the OLD code for robustness, 
#             # though this is still tricky due to line shifts. A simpler, common 
#             # approach is to assume the same function names were affected.
#             # We skip detailed old line mapping here for brevity and focus on the main win:
#             # analyzing smaller blocks of code.
            
#             # For simplicity, we analyze the old version of the file and the functions identified by the new lines.
#             old_functions = get_functions_in_diff_range(old_code, changed_lines_new)

        
#         # Use the union of new and old functions to calculate delta
        
#         new_total_ast = sum(count_ast_nodes(f) for f in new_functions)
#         old_total_ast = sum(count_ast_nodes(f) for f in old_functions)
        
#         new_total_complexity = sum(get_complexity(f) for f in new_functions)
#         old_total_complexity = sum(get_complexity(f) for f in old_functions)

#         ast_delta += abs(new_total_ast - old_total_ast)
#         complexity_delta += abs(new_total_complexity - old_total_complexity)

#         # Max function change calculation (can be simplified if only looking at snippets)
#         if new_functions:
#             max_func_change = max(max_func_change, max(len(f.splitlines()) for f in new_functions))
            
#         # ast_delta += abs(count_ast_nodes(new_code) - count_ast_nodes(old_code))
#         # complexity_delta += abs(get_complexity(new_code) - get_complexity(old_code))
#         # max_func_change = max(max_func_change, len(new_code.splitlines()))
    
#     return {
#         "ast_delta": ast_delta,
#         "complexity_delta": complexity_delta,
#         "max_func_change": max_func_change
#     }

import ast
import re
import warnings
from typing import Dict, Iterable, List, Set, Tuple

from radon.complexity import cc_visit


def count_ast_nodes(code: str) -> int:
    """Count AST nodes in a Python code snippet."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
        return len(list(ast.walk(tree)))
    except Exception:
        return 0


def get_complexity(code: str) -> int:
    """Compute summed cyclomatic complexity for a Python code snippet."""
    try:
        blocks = cc_visit(code)
        return sum(block.complexity for block in blocks)
    except Exception:
        return 0


def _safe_end_lineno(node: ast.AST) -> int:
    """Return a safe end line number for a node."""
    end_lineno = getattr(node, "end_lineno", None)
    lineno = getattr(node, "lineno", None)

    if isinstance(end_lineno, int):
        return end_lineno
    if isinstance(lineno, int):
        return lineno
    return 1


def _qualified_function_name(node: ast.AST, parents: Dict[int, ast.AST]) -> str:
    """
    Build a stable-ish function identifier, including class nesting when available.
    Examples:
      foo
      MyClass.bar
      outer.inner
    """
    parts = []
    current = node

    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(current.name)
        current = parents.get(id(current))

    return ".".join(reversed(parts))


def extract_functions_by_changed_lines(code: str, changed_lines: Set[int]) -> Dict[str, str]:
    """
    Return affected functions as {qualified_name: source_code}.
    A function is affected if any changed line intersects its span.
    """
    if not code or not changed_lines:
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except Exception:
        return {}

    lines = code.splitlines()
    parents: Dict[int, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    functions: Dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start_line = getattr(node, "lineno", 1)
        end_line = _safe_end_lineno(node)

        # Exact overlap test
        if changed_lines.isdisjoint(range(start_line, end_line + 1)):
            continue

        func_name = _qualified_function_name(node, parents)
        func_source = "\n".join(lines[start_line - 1 : end_line])
        functions[func_name] = func_source

    return functions


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@",
    re.MULTILINE,
)


def parse_changed_lines_from_patch(patch_text: str) -> Tuple[Set[int], Set[int]]:
    """
    Parse exact changed line numbers from a unified diff patch.

    Returns:
        (changed_old_lines, changed_new_lines)

    Notes:
    - Tracks only actual removed/addded lines.
    - Context lines advance both counters.
    - Ignores file headers ('---', '+++').
    """
    changed_old: Set[int] = set()
    changed_new: Set[int] = set()

    if not patch_text:
        return changed_old, changed_new

    lines = patch_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        m = _HUNK_RE.match(line)
        if not m:
            i += 1
            continue

        old_line = int(m.group("old_start"))
        new_line = int(m.group("new_start"))
        i += 1

        while i < len(lines):
            hunk_line = lines[i]

            if _HUNK_RE.match(hunk_line):
                break

            # Ignore diff file headers inside patch text
            if hunk_line.startswith("---") or hunk_line.startswith("+++"):
                i += 1
                continue

            if hunk_line.startswith("\\ No newline at end of file"):
                i += 1
                continue

            if hunk_line.startswith("+"):
                changed_new.add(new_line)
                new_line += 1
            elif hunk_line.startswith("-"):
                changed_old.add(old_line)
                old_line += 1
            else:
                # context line (' ' or anything else treated as context)
                old_line += 1
                new_line += 1

            i += 1

    return changed_old, changed_new


def _decode_blob(blob) -> str:
    """Safely decode a GitPython blob to text."""
    if blob is None:
        return ""
    try:
        return blob.data_stream.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _decode_patch(diff_obj) -> str:
    """Safely decode a GitPython diff patch to text."""
    try:
        if diff_obj.diff is None:
            return ""
        return diff_obj.diff.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_code_structural_features(diff_text: Iterable) -> Dict[str, int]:
    """
    Compute approximate structural change features from a Git diff.

    Features:
    - ast_delta: summed absolute AST-node delta across affected functions
    - complexity_delta: summed absolute cyclomatic-complexity delta across affected functions
    - max_func_change: size in lines of the largest affected function (old or new)

    Important:
    This is still an approximation. It compares functions whose spans intersect
    exact changed lines extracted from hunks.
    """
    ast_delta = 0
    complexity_delta = 0
    max_func_change = 0

    if diff_text is None:
        return {
            "ast_delta": 0,
            "complexity_delta": 0,
            "max_func_change": 0,
        }

    for d in diff_text:
        path = d.b_path or d.a_path
        if not path or not path.endswith(".py"):
            continue

        patch_text = _decode_patch(d)
        if not patch_text:
            continue

        old_code = _decode_blob(d.a_blob)
        new_code = _decode_blob(d.b_blob)

        changed_old_lines, changed_new_lines = parse_changed_lines_from_patch(patch_text)

        if not changed_old_lines and not changed_new_lines:
            continue

        old_functions = extract_functions_by_changed_lines(old_code, changed_old_lines)
        new_functions = extract_functions_by_changed_lines(new_code, changed_new_lines)

        all_function_names = set(old_functions) | set(new_functions)

        for func_name in all_function_names:
            old_func = old_functions.get(func_name, "")
            new_func = new_functions.get(func_name, "")

            old_ast = count_ast_nodes(old_func)
            new_ast = count_ast_nodes(new_func)

            old_complexity = get_complexity(old_func)
            new_complexity = get_complexity(new_func)

            ast_delta += abs(new_ast - old_ast)
            complexity_delta += abs(new_complexity - old_complexity)

            max_func_change = max(
                max_func_change,
                len(old_func.splitlines()) if old_func else 0,
                len(new_func.splitlines()) if new_func else 0,
            )

    return {
        "ast_delta": int(ast_delta),
        "complexity_delta": int(complexity_delta),
        "max_func_change": int(max_func_change),
    }