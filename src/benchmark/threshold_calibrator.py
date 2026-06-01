"""
Threshold calibration: trains a binary classifier on (query_A, query_B, should_cache)
pairs, replacing the fixed cosine threshold with a learned one per query type.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib


@dataclass
class CalibrationSample:
    embedding_a: list[float]
    embedding_b: list[float]
    query_type: Literal["factual", "creative", "code", "default"]
    should_cache: bool


class ThresholdCalibrator:
    def __init__(self):
        self.models: dict[str, LogisticRegression] = {}

    def _features(self, a: list[float], b: list[float]) -> np.ndarray:
        va, vb = np.array(a), np.array(b)
        cosine = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        l2 = float(np.linalg.norm(va - vb))
        dot = float(np.dot(va, vb))
        return np.array([cosine, l2, dot])

    def train(self, samples: list[CalibrationSample]):
        by_type: dict[str, list] = {}
        for s in samples:
            by_type.setdefault(s.query_type, []).append(s)

        for qtype, type_samples in by_type.items():
            X = np.array([self._features(s.embedding_a, s.embedding_b) for s in type_samples])
            y = np.array([int(s.should_cache) for s in type_samples])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(class_weight="balanced")
            clf.fit(X_train, y_train)

            print(f"\n[{qtype}] Classification report:")
            print(classification_report(y_test, clf.predict(X_test)))
            self.models[qtype] = clf

    def should_cache(self, a: list[float], b: list[float], query_type: str = "default") -> bool:
        model = self.models.get(query_type, self.models.get("default"))
        if model is None:
            # fallback to cosine threshold
            va, vb = np.array(a), np.array(b)
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))) >= 0.93
        feats = self._features(a, b).reshape(1, -1)
        return bool(model.predict(feats)[0])

    def save(self, path: str = "models/calibrator.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)

    def load(self, path: str = "models/calibrator.joblib"):
        self.models = joblib.load(path)
