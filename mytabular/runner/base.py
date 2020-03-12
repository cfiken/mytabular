from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import NamedTuple, List, Dict, Tuple
import numpy as np
import pandas as pd


class Result(NamedTuple):
    oof_preds: np.ndarray
    models: List
    metrics: Dict


class BaseRunner(metaclass=ABCMeta):

    def __init__(self, name: str, description: str = '') -> None:
        self.name = name
        self.description = description
        self.feature_names: List[str] = []
        self.outputdir = Path('./mytabular/output') / self.name
        if not self.outputdir.exists():
            self.outputdir.mkdir()

    def submit(self, pred: pd.DataFrame) -> None:
        pred.to_csv(self.outputdir / 'submission.csv', index=False)

    @abstractmethod
    def create_feature(self, *args, **kwargs):
        raise NotImplementedError()

    def _feature_wrapper(self, feature_name_tuple: Tuple[pd.DataFrame, str]) -> pd.DataFrame:
        feature, name = feature_name_tuple
        self.feature_names.append(name)
        return feature

    @abstractmethod
    def split_validation(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def notify(self):
        raise NotImplementedError()
