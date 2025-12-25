import ast
import warnings
from radon.complexity import cc_visit
import re

def count_ast_nodes(code):
    try:
        # tree = ast.parse(code)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
        return len(list(ast.walk(tree)))
    except Exception:
        return 0


def get_complexity(code):
    try:
        blocks = cc_visit(code)
        return sum(b.complexity for b in blocks)
    except Exception:
        return 0

def get_functions_in_diff_range(code: str, changed_lines: set) -> list[str]:
    """
    Identifies and extracts the full source code of functions that intersect
    with the set of lines changed in the diff.
    """
    functions_to_analyze = []
    
    # 1. Parse the AST to find function/class definitions and their line numbers
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    # 2. Iterate over all nodes to find function/method definitions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The function definition starts at node.lineno (1-based)
            # Find the end line number (requires using the source code or a helper)
            # For simplicity, we assume end is the last line of the function block.
            # A perfect solution would use astor or similar tools, but we approximate:
            
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line + 1) # Fallback if no end_lineno
            
            # Check if the function body (including its signature) overlaps with changed lines
            if any(line in changed_lines for line in range(start_line, end_line + 1)):
                
                # Extract the source code of the function body
                lines = code.splitlines()
                # ast lines are 1-based, list indices are 0-based
                func_lines = lines[start_line - 1 : end_line] 
                functions_to_analyze.append('\n'.join(func_lines))
    
    return functions_to_analyze

def extract_code_structural_features(diff_text):
    ast_delta = 0
    complexity_delta = 0
    max_func_change = 0

    for d in diff_text:
        if not d.b_path or not d.b_blob:
            continue

        # 💡 Optimization Step 1: Skip non-Python files (optional but highly recommended)
        if not d.b_path.endswith('.py'):
            continue

        try:
            new_code = d.b_blob.data_stream.read().decode('utf-8', errors='ignore')
            old_code = d.a_blob.data_stream.read().decode('utf-8', errors='ignore') if d.a_blob else ""
        except Exception:
            continue

        changed_lines_new = set()

        patch_text = d.diff.decode(errors="ignore")
        
        # This is a basic way to get lines. A more robust way uses the diff library.
        # It relies on reading the hunk headers (@@ -old_start,old_count +new_start,new_count @@)
        
        for hunk_match in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch_text, re.MULTILINE):
            new_start = int(hunk_match.group(1))
            new_count = int(hunk_match.group(2) or 1)
            changed_lines_new.update(range(new_start, new_start + new_count))


        # 💡 Optimization Step 3: Analyze only the code of affected functions
        
        # Analyze NEW code
        new_functions = get_functions_in_diff_range(new_code, changed_lines_new)
        
        # Analyze OLD code (Need to find the corresponding old lines, which is complex. 
        # For simplicity in this fix, we analyze the *new* versions and the *old* versions
        # corresponding to the *newly identified* functions.)
        old_functions = []
        if old_code:
            # Re-run changed lines identification on the OLD code for robustness, 
            # though this is still tricky due to line shifts. A simpler, common 
            # approach is to assume the same function names were affected.
            # We skip detailed old line mapping here for brevity and focus on the main win:
            # analyzing smaller blocks of code.
            
            # For simplicity, we analyze the old version of the file and the functions identified by the new lines.
            old_functions = get_functions_in_diff_range(old_code, changed_lines_new)

        
        # Use the union of new and old functions to calculate delta
        
        new_total_ast = sum(count_ast_nodes(f) for f in new_functions)
        old_total_ast = sum(count_ast_nodes(f) for f in old_functions)
        
        new_total_complexity = sum(get_complexity(f) for f in new_functions)
        old_total_complexity = sum(get_complexity(f) for f in old_functions)

        ast_delta += abs(new_total_ast - old_total_ast)
        complexity_delta += abs(new_total_complexity - old_total_complexity)

        # Max function change calculation (can be simplified if only looking at snippets)
        if new_functions:
            max_func_change = max(max_func_change, max(len(f.splitlines()) for f in new_functions))
            
        # ast_delta += abs(count_ast_nodes(new_code) - count_ast_nodes(old_code))
        # complexity_delta += abs(get_complexity(new_code) - get_complexity(old_code))
        # max_func_change = max(max_func_change, len(new_code.splitlines()))
    
    return {
        "ast_delta": ast_delta,
        "complexity_delta": complexity_delta,
        "max_func_change": max_func_change
    }