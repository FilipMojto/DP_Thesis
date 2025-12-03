import git
from git import Commit
from pathlib import Path
from src_code.config import PYTHON_LIBS_DIR
from typing import Dict
# Import code analysis tools (e.g., radon, pygount)

class FeatureExtractor:
    # def __init__(self, repo_url_map: dict):
    #     self.repo_url_map = repo_url_map
    #     # Initialize code embedding model/tokenizer here
    def __init__(self, repo_url_map: dict):
        self.repo_url_map = repo_url_map
        # self.repos:  = {} # Dictionary to store active GitPython Repo objects
        self.repos: Dict[str, git.Repo] = {}  # Dictionary to store active GitPython Repo objects
        self._initialize_repos() # NEW: Call initialization here

    def _clone_or_pull(self, repo_name, repo_url) -> Path:
        """Clones the repository if it doesn't exist, otherwise pulls."""
        repo_path = PYTHON_LIBS_DIR / repo_name
        if not repo_path.exists():
            print(f"[GIT] Cloning {repo_url} into {repo_path}")
            git.Repo.clone_from(repo_url, repo_path)
        else:
            print(f"[GIT] Updating {repo_name}")
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        return repo_path
    
    def _initialize_repos(self):
        """Clones all required repositories and stores the Repo object."""
        for repo_name, repo_url in self.repo_url_map.items():
            repo_path = self._clone_or_pull(repo_name, repo_url)
            self.repos[repo_name] = git.Repo(repo_path)
        
        print(f"[GIT] Initialization complete. Loaded repos: {list(self.repos.keys())}")

    def extract_features(self, repo_name: str, commit_hash: str) -> dict:
        """Extracts all specified metrics for a single commit."""
        # repo_url = self.repo_url_map.get(repo_name)
        # if not repo_url:
        #     return {}  # Handle missing repo URL

        # repo_path = self._clone_or_pull(repo_name, repo_url)
        # repo = git.Repo(repo_path)
        # commit = repo.commit(commit_hash)
        # 💡 FIX: Get the already loaded Repo object directly from the cache
        repo = self.repos.get(repo_name)
        if not repo:
            print(f"[ERROR] Repo {repo_name} not initialized.")
            return {}

        try:
            commit = repo.commit(commit_hash)
        except Exception as e:
            print(f"[ERROR] Commit {commit_hash} not found in repo {repo_name}: {e}")
            return {}
        
        # --- Check for Parent Commit ---
        if commit.parents:
            # Commit has parents, calculate diff relative to the first parent (normal case)
            parent_commit = commit.parents[0]
            diff = commit.diff(parent_commit)
            
            # stats is calculated relative to the *parent* by default for non-merge commits
            stats = commit.stats.total 
        else:
            # Commit is the initial commit (no parents)
            print(f"[WARN] Commit {commit_hash[:7]} in {repo_name} is the initial commit (no parents).")
            
            # Calculate diff relative to an empty tree (None), which represents all additions
            diff = commit.diff(git.NULL_TREE) 
            
            # stats calculation relative to an empty tree
            # stats.total will correctly show all files/lines as additions
            stats = commit.stats.total

        # 1. Change / Churn Metrics (loc added, deleted, files, hunks)
        stats = commit.stats.total
        features = {
            "loc_added": stats.get('insertions', 0),
            "loc_deleted": stats.get('deletions', 0),
            "files_changed": stats.get('files', 0),
            # "hunks_count": self._calculate_hunks(commit.diff(commit.parents[0])),
            "hunks_count": self._calculate_hunks(diff), # Use the calculated 'diff' object
        }

        # 2. Code Structural Metrics (complexity delta, max func change size)
        # Requires analyzing diffs using a tool like radon/ast libraries
        features.update(self._analyze_structural_changes(commit))

        # 3. Semantic / Embeddings (code embed, msg embed)
        # Requires running a CodeBERT/Code2Vec model on the diff and message
        features["msg_embed"] = self._get_message_embedding(commit.message)
        features["code_embed"] = self._get_code_embedding(commit)

        # 4. Textual and NLP features
        features["msg_len"] = len(commit.message)
        features["has_fix_kw"] = 1 if 'fix' in commit.message.lower() else 0
        
        # 5. Developer / Social and Historical Features
        # Requires iterating over the commit history of the author/file
        features.update(self._calculate_author_and_temporal(commit, repo))

        # 6. Line-level specific (Placeholder: Requires line-level diff processing)
        features["line_context_embed"] = [0.0] # Placeholder
        features["line_token_features"] = 0 # Placeholder: Count 'TODO', 'FIXME' etc.
        
        return features

    # --- Helper methods for complex calculations (must be implemented) ---
    def _calculate_hunks(self, diff):
        # Implementation to count hunks in the diff
        return sum(len(d.diff.decode('utf-8').split('@@')) - 1 for d in diff if d.diff) / 2
    
    def _analyze_structural_changes(self, commit: Commit) -> dict:
        """
        Calculates AST Node Delta and (simplified) complexity change.
        NOTE: A robust solution requires analyzing AST/complexity on both
              the parent and current commit versions of the modified files.
        """
        if commit.parents:
            parent_commit_sha = commit.parents[0].hexsha
        else:
            # Use a string representing the empty tree for the initial commit
            parent_commit_sha = '4b825dc642cb6eb9a060e54bf8d69288fbee4904' # SHA-1 of empty tree
            
        repo = commit.repo # Get the GitPython repo object from the commit

        ast_delta = 0
        max_func_change_loc = 0
        complexity_delta = 0
        repo = commit.repo # Get the GitPython repo object from the commit
        
        # Simplified Structural Analysis (Placeholder)
        # diff_text = commit.diff(commit.parents[0], create_patch=True).diff.decode('utf-8', errors='ignore')
        # 💡 FIX: Use the Git command interface to get the raw patch text
        # This executes 'git diff <parent_sha> <current_sha>' and returns the output string.
        try:
            diff_text = repo.git.diff(
                parent_commit_sha,
                commit.hexsha,
                patch=True, # Ensure it outputs a patch format
                no_color=True,
                ignore_errors=True # Optional, helps prevent crashing on weird data
            )
        except git.exc.GitCommandError as e:
            print(f"[WARN] Failed to get patch for {commit.hexsha[:7]}: {e}")
            diff_text = "" # Use empty string if diff fails
        
        # Ast Node Delta (Simplified: counting keywords added/removed)
        ast_delta += diff_text.count('\n+def ') - diff_text.count('\n-def ')
        ast_delta += diff_text.count('\n+class ') - diff_text.count('\n-class ')

        # Max Func Change Size (Simplified: assume largest change corresponds to a function)
        # In a real scenario, you'd track LOC per function body in the diff
        loc_changes = [len(l) for l in diff_text.splitlines() if l.startswith('+') or l.startswith('-')]
        if loc_changes:
             max_func_change_loc = max(loc_changes)

        # Complexity Delta (Placeholder: Requires tool like radon/xenon)
        # complexity_delta = self._calculate_cyclomatic_delta(commit) 

        return {
            "ast_node_delta": ast_delta,
            "max_func_change_size": max_func_change_loc,
            "complexity_delta": complexity_delta, # Currently 0, needs external tool
        }

    def _get_message_embedding(self, message: str):
        """
        Placeholder for generating semantic embedding of the commit message.
        In a real scenario, this involves a pre-trained Transformer model.
        """
        # Placeholder: Return a fixed-size list of random floats or just a stub value
        # e.g., using a SHA-256 hash as a deterministic stub:
        import hashlib
        hash_value = hashlib.sha256(message.encode()).hexdigest()
        return [float(int(hash_value[i:i+2], 16)) / 255.0 for i in range(0, 16, 2)] # 8-dim stub

    def _get_code_embedding(self, commit: Commit):
        """
        Placeholder for generating semantic embedding of the diff code.
        In a real scenario, this involves a CodeBERT/Code2Vec model.
        """
        # Get the full diff patch text
        diff = commit.diff(commit.parents[0], create_patch=True)
        code_text = diff.diff.decode('utf-8', errors='ignore')

        # Placeholder: Return a fixed-size list of random floats
        import hashlib
        hash_value = hashlib.sha256(code_text.encode()).hexdigest()
        return [float(int(hash_value[i:i+2], 16)) / 255.0 for i in range(16, 32, 2)] # 8-dim stub

    def _calculate_author_and_temporal(self, commit: Commit, repo: git.Repo) -> dict:
        """
        Calculates author experience, activity, and historical file churn metrics.
        This requires iterating over the commit history.
        """
        
        # --- Developer / Social Metrics ---
        
        # Author Experience (Total prior commits by author)
        author_email = commit.author.email
        author_exp = 0
        
        # Iterate over all prior commits in the *entire* repository history
        # NOTE: This can be slow, caching/pre-indexing is key for production!
        for c in repo.iter_commits(rev=f'{commit.hexsha}^', max_count=10000): # Limit history to speed up
            if c.author.email == author_email:
                author_exp += 1
        
        # Author Recent Activity (Commits in last 30 days)
        # Simplified calculation: not traversing history, relies on dataset pre-processing
        # In a real system, you'd filter the above iteration by date.
        
        # --- Historical / Temporal Features ---
        
        recent_churn = 0 # Sum of LOC changed in last K commits touching same file
        time_since_last_change = 0 # Time elapsed since previous modification of same file (in seconds)
        
        # Simplified: Use Git's 'log' command for efficiency on file history
        try:
            # Get the list of modified files
            file_paths = list(commit.stats.files.keys())
            if file_paths:
                # Target one file for simplicity in this example
                target_file = file_paths[0] 
                
                # Time since last change (using one file as proxy)
                # Find the previous commit that touched this file
                previous_commit = next(repo.iter_commits(rev=f'{commit.hexsha}^', paths=[target_file], max_count=1))
                time_since_last_change = int(commit.authored_date - previous_commit.authored_date)
                
                # Recent Churn (LOC changed in last 5 commits touching the file)
                churn_commits = repo.iter_commits(rev=f'{previous_commit.hexsha}^', paths=[target_file], max_count=5)
                for c in churn_commits:
                    recent_churn += c.stats.total.get('insertions', 0) + c.stats.total.get('deletions', 0)

        except StopIteration:
            pass # No prior history for the file
        except Exception as e:
            # print(f"Error calculating temporal features: {e}")
            pass
            
        return {
            "author_exp": author_exp,
            "author_recent_activity": 0, # Placeholder for complexity
            "recent_churn": recent_churn,
            "time_since_last_change": time_since_last_change,
        }