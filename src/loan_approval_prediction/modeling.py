"""Leakage-safe preprocessing and small candidate model set."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from loan_approval_prediction.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(estimator: object) -> Pipeline:
    """Fit imputers and encoders inside each training or cross-validation fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def candidate_models(random_state: int) -> dict[str, Pipeline]:
    """Return interpretable and tree-based baselines with modest fixed settings."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=2000, random_state=random_state
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            )
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=1,
                random_state=random_state,
            )
        ),
    }
