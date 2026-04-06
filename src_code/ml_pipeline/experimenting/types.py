
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, TypeAlias
from pydantic import BaseModel, Field

from src_code.config import SubsetType, SupportedModel


Config = Mapping[str, Any]

ARG_RESOLVER: TypeAlias = Callable[[Config], Any]
ARG_VALIDATOR: TypeAlias = Callable[[Config], None]

ARG_RESOLVERS_COLL: TypeAlias = Dict[str, ARG_VALIDATOR]
ARG_VALIDATORS_COLL: TypeAlias = List[ARG_VALIDATOR]

SubsetArg = Literal['train', 'test', 'val', 'all']


class Artifact(BaseModel):
    ...

class MyDataset(BaseModel):
    type: SubsetType
    
    rows: Optional[int] = None
    cols: Optional[int] = None
    src_path: Optional[Path] = None
    # rows_after: Optional[int] = None
    # cols_after: Optional[int] = None


class EdaResults(BaseModel):
    loaded_datasets: List[MyDataset] = Field(default_factory=list)
    EDA_ready_datasets: List[MyDataset] = Field(default_factory=list)


class TuningResults(BaseModel):
    param_artifact: Optional[Path] = None
    features_trained_on: Optional[int] = None


class PreprocessingResults(BaseModel):
    # col_before: Optional[int] = None
    # col_after: Optional[int] = None
    # row_before: Optional[int] = None
    # row_after: Optional[int] = None
    loaded_datasets: List[MyDataset] = Field(default_factory=list)
    preprocessed_datasets: List[MyDataset] = Field(default_factory=list)
    # engineered_cols: Optional[int] = None


class TransformationResults(BaseModel):
    pass

class TrainingResults(BaseModel):
    tuning_params: Optional[Path] = None
    cv_scores: Optional[Dict[str, float]] = None
    trained_model: Optional[Path] = None


class EvalResults(BaseModel):
    model: SupportedModel
    best_thresh_f2: Optional[float] = None
    best_f2_score: Optional[float] = None
    roc_auc: Optional[float] = None
    auprc: Optional[float] = None


# @dataclass
class Experiment(BaseModel):
    # @staticmethod
    # def generate_id():
    #     return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- General Info ---
    experiment_id: str = Field(default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    date: datetime = Field(default_factory=datetime.now)
    # models: List[SupportedModel] = field(default_factory=list)
    models: Optional[List[SupportedModel]] = None
    notes: Optional[str] = None
    is_finished: bool = Field(default=False)

    # --- Data Info ---
    training_subset: Optional[Path] = None
    testing_subset: Optional[Path] = None
    validation_subset: Optional[Path] = None

    # --- Phase Results ---
    # eval_results: List[EvalResults] = Field(default_factory=list)
    # Phase results should probably be Optional or have default factories 
    # so you can create the object before the phases are run
    eda_results: EdaResults = Field(default_factory=EdaResults)
    tuning_results: TuningResults = Field(default_factory=TuningResults)
    preprocessing_results: PreprocessingResults = Field(default_factory=PreprocessingResults)
    transformation_results: TransformationResults = Field(default_factory=TransformationResults)
    training_results: List[TrainingResults] = Field(default_factory=list)
    eval_results: List[EvalResults] = Field(default_factory=list)


# if __name__ == "__main__":
#     exp = Experiment()