
from sklearn.calibration import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from notebooks.constants import TARGET
from notebooks.logging_config import MyLogger
from src_code.config import LOG_DIR, MODEL_DIR, PROCESSED_DATA_DIR
from src_code.ml_pipeline.data_utils import load_df, load_model
from src_code.versioning import VersionedFileManager


if __name__ == "__main__":
    script_logger = MyLogger(label="test", section_name="test", file_log_path=LOG_DIR / "test.log")
    df_versioner = VersionedFileManager(file_path=PROCESSED_DATA_DIR / "train_engineered.feather", logger=script_logger)
    RF_versioner = VersionedFileManager(file_path=MODEL_DIR / "RF_model_train", logger=script_logger)

    df_train = load_df(df_file_path=df_versioner.current_newest, logger=script_logger)

    RF = load_model(path=RF_versioner.current_newest, logger=script_logger)

    # Separate features and target
    X_train = df_train.drop(columns=[TARGET])  # adjust "target" to your actual column
    y_train = df_train[TARGET]

    # Use StratifiedKFold to keep class distribution in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    script_logger.log_check("Starting cross-validation on training data...")

    # Get predictions for all folds
    y_pred = cross_val_predict(RF, X_train, y_train, cv=skf, n_jobs=-1)

    # Classification report
    report = classification_report(y_train, y_pred, output_dict=False)
    script_logger.log_result("Cross-validation results on training set:")
    script_logger.log_result(report)

