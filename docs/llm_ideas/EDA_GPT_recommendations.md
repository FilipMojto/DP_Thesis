Example from your EDA

Preprocessing:

- Normalize loc_added, loc_deleted using log1p to reduce skew.

- Cap extreme outliers in recent_churn to prevent distortions.

- Ensure time_since_last_change is non-negative (or handle negative values).

Feature Engineering:

- Ratio: loc_added / files_changed → how dense are commits.

- Interaction: author_exp_pre * author_recent_activity_pre → experience × activity.

- Aggregate: total AST change per author in last 30 days.

- Embedding metrics: norm(code_embed) or cosine similarity between commit message and code embeddings.

2️⃣ Full Preprocessing + Feature Engineering Plan (Based on Your EDA)
A. Preprocessing

1. Missing values

    - Keep only rows with minimal missing values, or impute with median/mean.

    - Log zero or near-zero proportions for reference.

2. Outliers

    - For numeric features with >10% outliers (e.g., ast_delta, complexity_delta, recent_churn), use:

        - Capping (e.g., 1st–99th percentile)

        - Or log transform if strictly positive (log1p)

3. Skewed numeric features

    - Apply log1p or Box-Cox/Yeo-Johnson to reduce skew.

    - Candidate features: loc_added, loc_deleted, files_changed, hunks_count, recent_churn, ast_delta.

4. Time-related features

    - Fix negative time_since_last_change or replace with absolute value.

    - Scale in days or weeks for stability.

5. Embeddings

    - Compute norms (np.linalg.norm) as scalar features.

    - Optionally reduce dimensions via PCA (already tried).

B. Feature Engineering

Interaction features

author_exp_pre * author_recent_activity_pre → experience × recent activity.

loc_added * files_changed → commit density.

Ratio features

loc_added / max(1, files_changed) → average lines per file.

ast_delta / max_func_change → average AST change per function.

Temporal aggregation

Average recent churn per author or repo.

Time since last bug fix normalized per repo.

Embedding features

Norms (already added).

Cosine similarity between code_embed and msg_embed.

Keyword counts

Sum or ratios: total_kw_count = todo + fixme + try + except + raise.

Zero/non-zero ratios to indicate “presence of special instructions”.

Label-guided feature analysis

Keep top correlated features (loc_added, hunks_count, etc.).

Consider thresholds for highly predictive outliers (e.g., massive churn).

C. Scaling / Encoding

Numeric features: standardize or normalize after transformations.

Categorical features: one-hot encode small cardinality (repo, ext).

Embeddings: either use raw vectors (with neural nets) or reduce via PCA/TSNE/UMAP.

D. Optional: Dimensionality Reduction

Use PCA on embeddings to 2–5 dimensions for tree models.

Use UMAP or t-SNE for visualization or exploratory clustering.