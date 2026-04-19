# -----------------------------------------------------------------------------
# TEMPORARY FILE: contains old logic for risk-based assessment of ml models and baselines
# about to be Deleted soon
# -----------------------------------------------------------------------------



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from notebooks.logging_config import MyLogger
# from src_code.config import ENGINEERED_DATA_DIR, MODEL_DIR, REPORTS_DIR, SupportedModel
# import src_code.ml_pipeline.data_utils as dutls
# from src_code.ml_pipeline.preprocessing.feature_config import DROP_COLS
# from src_code.ml_pipeline.preprocessing.preprocessing import drop_cols
# from src_code.versioning import VersionedFileManager

# logger = MyLogger(
#     label="BASELINE_EVAL",
#     section_name="BASELINE EVALUATION LOGGER",
#     file_log_path=MODEL_DIR / "baseline_eval_log.log",
# )

# model_type: SupportedModel = 'XGB'

# # ---------------- LOAD MODEL ----------------
# model = dutls.load_artifact(
#     dir=MODEL_DIR,
#     artifact_type="trained_model",
#     logger=logger,
#     label=model_type,
# )

# # ---------------- LOAD DATA ----------------
# test_df_versioner = VersionedFileManager(
#     file_path=ENGINEERED_DATA_DIR / "test_engineered.feather",
#     logger=logger,
# )

# test_df = dutls.load_df(
#     df_file_path=test_df_versioner.current_newest, logger=logger
# )

# test_df = drop_cols(df=test_df, cols=DROP_COLS, logger=logger)

# # ---------------- PREPARE DATA ----------------
# model_wrapper = model.model_wrapper
# X_trans = model_wrapper.transform(test_df)

# y_true = test_df["label"].values
# y_proba = model_wrapper.model.predict_proba(X_trans)[:, 1]

# # ---------------- SORT BY RISK ----------------
# df_sorted = pd.DataFrame({
#     "y_proba": y_proba,
#     "y_true": y_true
# }).sort_values(by="y_proba", ascending=False).reset_index(drop=True)

# total_bugs = df_sorted["y_true"].sum()

# # ---------------- EFFORT SIMULATION ----------------
# effort_levels = np.linspace(0.01, 1.0, 50)  # 1% → 100%
# recalls = []

# for effort in effort_levels:
#     k = int(effort * len(df_sorted))
#     top_k = df_sorted.iloc[:k]
    
#     bugs_found = top_k["y_true"].sum()
#     recall_at_k = bugs_found / total_bugs if total_bugs > 0 else 0
    
#     recalls.append(recall_at_k)

# # ---------------- BASELINE (RANDOM) ----------------
# # random expectation = linear
# baseline_recalls = effort_levels.copy()

# # ---------------- OPTIONAL: PRINT FEW POINTS ----------------
# for e, r in zip([0.1, 0.3, 0.5], 
#                 [recalls[int(0.1*49)], recalls[int(0.3*49)], recalls[int(0.5*49)]]):
#     print(f"Effort {int(e*100)}% -> Bugs found: {r:.2f}")


# # ---------------- HEURISTIC FEATURES ----------------

# # Create loc_change if not present
# if "loc_change" not in test_df.columns:
#     if "loc_added" in test_df.columns and "loc_deleted" in test_df.columns:
#         test_df["loc_change"] = test_df["loc_added"] + test_df["loc_deleted"]
#     else:
#         raise ValueError("loc_added / loc_deleted not found for heuristic")

# # Choose heuristic columns
# heuristics = {
#     "LOC_CHANGE": test_df["loc_change"],
# }

# if "files_changed" in test_df.columns:
#     heuristics["FILES_CHANGED"] = test_df["files_changed"]

# # ---------------- FUNCTION FOR EFFORT CURVE ----------------

# def compute_effort_curve(scores, y_true):
#     df = pd.DataFrame({
#         "score": scores,
#         "y_true": y_true
#     }).sort_values(by="score", ascending=False).reset_index(drop=True)

#     total_bugs = df["y_true"].sum()
    
#     effort_levels = np.linspace(0.01, 1.0, 50)
#     recalls = []

#     for effort in effort_levels:
#         k = max(1, int(effort * len(df)))
#         top_k = df.iloc[:k]

#         bugs_found = top_k["y_true"].sum()
#         recall_at_k = bugs_found / total_bugs if total_bugs > 0 else 0

#         recalls.append(recall_at_k)

#     return effort_levels, recalls

# # ---------------- COMPUTE CURVES ----------------

# # ML curve (already computed, reuse or recompute cleanly)
# effort_levels, ml_recalls = compute_effort_curve(y_proba, y_true)

# # Heuristic curves
# heuristic_results = {}
# for name, scores in heuristics.items():
#     _, recalls = compute_effort_curve(scores.values, y_true)
#     heuristic_results[name] = recalls

# # Baseline
# baseline_recalls = effort_levels.copy()

# # ---------------- PLOT ----------------

# plt.figure()

# # ML
# plt.plot(effort_levels, ml_recalls, label=f"ML Model ({model_type})", linewidth=2)

# # Heuristics
# for name, recalls in heuristic_results.items():
#     plt.plot(effort_levels, recalls, linestyle="--", label=f"Heuristic ({name})")

# # Baseline
# plt.plot(effort_levels, baseline_recalls, linestyle=":", label="Random baseline")

# plt.xlabel("Percentage of Commits Inspected")
# plt.ylabel("Percentage of Bugs Found (Recall)")
# plt.title("Effort-based Comparison: ML vs Heuristics")
# plt.legend()
# plt.grid()

# TARGET_DIR = REPORTS_DIR / "effort_curves"
# if not TARGET_DIR.exists():
#     TARGET_DIR.mkdir(parents=True, exist_ok=True)

# plt.savefig(TARGET_DIR / "effort_curve_comparison.png")
# plt.show()

