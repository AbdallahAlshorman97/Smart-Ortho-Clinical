from __future__ import annotations

from typing import Dict, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Age",
    "HR",
    "Temp",
    "VAS",
    "ODI",
    "Systolic_BP",
    "Diastolic_BP",
    "WBC",
    "CRP",
    "ESR",
    "Hemoglobin",
    "Labs_Available",
]
TEXT_FEATURES = ["Complaint", "Imaging", "Labs"]
CATEGORICAL_FEATURES = ["Sex"]


def _flatten_text_column(values):
    return values.astype(str).ravel()


def build_text_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=""),
            ),
            (
                "flatten",
                FunctionTransformer(_flatten_text_column, validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
        ]
    )


def build_preprocessor(
    include_complaint: bool = True,
    include_imaging: bool = True,
    include_labs_text: bool = True,
) -> ColumnTransformer:
    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        (
            "sex",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]

    if include_complaint:
        transformers.append(("complaint", build_text_pipeline(), ["Complaint"]))
    if include_imaging:
        transformers.append(("imaging", build_text_pipeline(), ["Imaging"]))
    if include_labs_text:
        transformers.append(("labs", build_text_pipeline(), ["Labs"]))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def build_model_pipeline(
    classifier_name: str = "logistic",
    include_complaint: bool = True,
    include_imaging: bool = True,
    include_labs_text: bool = True,
) -> Pipeline:
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    preprocessor = build_preprocessor(
        include_complaint=include_complaint,
        include_imaging=include_imaging,
        include_labs_text=include_labs_text,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_model_config(include_complaint: bool, include_imaging: bool, include_labs_text: bool) -> Dict[str, object]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if include_complaint:
        features.append("Complaint")
    if include_imaging:
        features.append("Imaging")
    if include_labs_text:
        features.append("Labs")
    return {"feature_columns": features}