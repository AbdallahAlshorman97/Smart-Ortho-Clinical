from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.feature_engineering import build_model_pipeline, get_model_config
from src.prepare_data import RISK_MAPPING, build_clean_dataframe, load_dataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = ROOT / "data" / "processed" / "03_model_ready_dataset.csv"
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "splits" / "validation.csv"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "smart_ortho_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORT_PATH = ROOT / "reports" / "model_results.md"
ERROR_ANALYSIS_PATH = ROOT / "reports" / "error_analysis.csv"


def load_or_prepare_data() -> pd.DataFrame:
    if not PROCESSED_DATA.exists():
        print("Processed dataset not found. Rebuilding from source workbook.")
        build_clean_dataframe(load_dataset(ROOT / "data" / "original" / "01_original_dataset.xlsx")).to_csv(PROCESSED_DATA, index=False)

    return pd.read_csv(PROCESSED_DATA)


def create_splits(df: pd.DataFrame) -> None:
    if TRAIN_CSV.exists() and VALIDATION_CSV.exists() and TEST_CSV.exists():
        return

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["target"],
        random_state=42,
    )
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["target"],
        random_state=42,
    )
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    validation.to_csv(VALIDATION_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print("Created train/validation/test splits")


def metrics_for_target(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    labels = [0, 1, 2]
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    if y_prob is not None and y_prob.shape[1] == len(labels):
        try:
            metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_prob, multi_class="ovo")
        except Exception:
            metrics["roc_auc_ovo"] = None
    else:
        metrics["roc_auc_ovo"] = None

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    per_class = {}
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        false_negative_rate = fn / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "false_negative_rate": false_negative_rate,
        }
    metrics["per_class"] = per_class
    return metrics


def evaluate_pipeline(
    name: str,
    pipeline_obj: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, Any]:
    target = "target"
    X_train = train_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_train = train_df[target]
    X_val = validation_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_val = validation_df[target]
    X_test = test_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_test = test_df[target]

    pipeline_obj.fit(X_train, y_train)
    y_val_pred = pipeline_obj.predict(X_val)
    y_val_prob = pipeline_obj.predict_proba(X_val)
    y_test_pred = pipeline_obj.predict(X_test)
    y_test_prob = pipeline_obj.predict_proba(X_test)

    results = {
        "name": name,
        "validation": metrics_for_target(y_val.to_numpy(), y_val_pred, y_val_prob),
        "test": metrics_for_target(y_test.to_numpy(), y_test_pred, y_test_prob),
        "model_type": type(pipeline_obj.named_steps["classifier"]).__name__,
        "feature_columns": feature_columns,
    }
    return results


def grouped_evaluation(pipeline_obj: Any, df: pd.DataFrame) -> Dict[str, Any]:
    group_label = "Complaint"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    groups = df[group_label]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train = train_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_test = test_df["target"]
    pipeline_obj.fit(X_train, y_train)
    y_pred = pipeline_obj.predict(X_test)
    y_prob = pipeline_obj.predict_proba(X_test)
    return metrics_for_target(y_test.to_numpy(), y_pred, y_prob)


def save_report(results: List[Dict[str, Any]], chosen: Dict[str, Any], grouped_results: Dict[str, Any], baseline_accuracy: float) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Smart-Ortho Model Results",
        "",
        f"Training date: {datetime.utcnow().isoformat()}Z",
        "",
        "## Selected model",
        "",
        f"- Name: {chosen['name']}",
        f"- Model type: {chosen['model_type']}",
        f"- Validation macro F1: {chosen['validation']['macro_f1']:.4f}",
        f"- Test macro F1: {chosen['test']['macro_f1']:.4f}",
        "",
        "## Baseline",
        "",
        f"- Majority class accuracy: {baseline_accuracy:.4f}",
        "",
        "## Experiment results",
        "",
    ]
    for result in results:
        lines.append(f"### {result['name']}")
        lines.append(f"- Model type: {result['model_type']}")
        lines.append(f"- Validation accuracy: {result['validation']['accuracy']:.4f}")
        lines.append(f"- Validation macro F1: {result['validation']['macro_f1']:.4f}")
        lines.append(f"- Test accuracy: {result['test']['accuracy']:.4f}")
        lines.append(f"- Test macro F1: {result['test']['macro_f1']:.4f}")
        lines.append("")
    lines.extend([
        "## Grouped evaluation",
        "",
        f"- Test accuracy: {grouped_results['accuracy']:.4f}",
        f"- Test macro F1: {grouped_results['macro_f1']:.4f}",
        "",
        "## Limitations",
        "",
        "- The dataset includes text fields likely to leak the final diagnosis into complaint and imaging descriptions.",
        "- Model B uses only pre-diagnostic structured features and the parsed laboratory results; it remains an experimental baseline.",
        "- The prototype is investigational and not clinically validated.",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {REPORT_PATH}")


def save_error_analysis(test_df: pd.DataFrame, pipeline_obj: Any) -> None:
    X_test = test_df.drop(columns=["ID", "dataset_label", "risk_level", "recommended_time", "target"])
    y_test = test_df["target"]
    y_pred = pipeline_obj.predict(X_test)
    y_prob = pipeline_obj.predict_proba(X_test)
    scores = np.max(y_prob, axis=1)
    errors = test_df.copy()
    errors["predicted_target"] = y_pred
    errors["prediction_confidence"] = scores
    errors["predicted_label"] = errors["predicted_target"].map({k: v["original_label"] for k, v in RISK_MAPPING.items()})
    errors["predicted_level"] = errors["predicted_target"].map({k: v["level"] for k, v in RISK_MAPPING.items()})
    errors = errors[[
        "ID",
        "target",
        "dataset_label",
        "predicted_target",
        "predicted_label",
        "prediction_confidence",
        "Complaint",
        "HR",
        "Systolic_BP",
        "Diastolic_BP",
        "Temp",
        "Imaging",
    ]]
    errors.to_csv(ERROR_ANALYSIS_PATH, index=False)
    print(f"Saved error analysis to {ERROR_ANALYSIS_PATH}")


def train_best_model() -> None:
    df = load_or_prepare_data()
    create_splits(df)
    train_df = pd.read_csv(TRAIN_CSV)
    validation_df = pd.read_csv(VALIDATION_CSV)
    test_df = pd.read_csv(TEST_CSV)

    experiments = [
        {
            "name": "Model A - Logistic Regression",
            "classifier": "logistic",
            "include_complaint": True,
            "include_imaging": True,
            "include_labs_text": True,
        },
        {
            "name": "Model A - Random Forest",
            "classifier": "random_forest",
            "include_complaint": True,
            "include_imaging": True,
            "include_labs_text": True,
        },
        {
            "name": "Model B - Logistic Regression",
            "classifier": "logistic",
            "include_complaint": False,
            "include_imaging": False,
            "include_labs_text": False,
        },
        {
            "name": "Model B - Random Forest",
            "classifier": "random_forest",
            "include_complaint": False,
            "include_imaging": False,
            "include_labs_text": False,
        },
        {
            "name": "Experiment B - No complaint",
            "classifier": "logistic",
            "include_complaint": False,
            "include_imaging": True,
            "include_labs_text": True,
        },
        {
            "name": "Experiment C - No complaint or imaging",
            "classifier": "logistic",
            "include_complaint": False,
            "include_imaging": False,
            "include_labs_text": True,
        },
    ]

    results = []
    best_result: Optional[Dict[str, Any]] = None
    best_pipeline = None
    model_comparison = []

    for experiment in experiments:
        pipeline = build_model_pipeline(
            classifier_name=experiment["classifier"],
            include_complaint=experiment["include_complaint"],
            include_imaging=experiment["include_imaging"],
            include_labs_text=experiment["include_labs_text"],
        )
        feature_columns = get_model_config(
            include_complaint=experiment["include_complaint"],
            include_imaging=experiment["include_imaging"],
            include_labs_text=experiment["include_labs_text"],
        )["feature_columns"]
        result = evaluate_pipeline(
            experiment["name"],
            pipeline,
            train_df,
            validation_df,
            test_df,
            feature_columns,
        )
        results.append(result)
        validation = result["validation"]
        test = result["test"]
        validation_per_class = validation.get("per_class", {})
        test_per_class = test.get("per_class", {})
        avg_validation_recall = sum(float(item.get("recall", 0.0)) for item in validation_per_class.values()) / max(len(validation_per_class), 1)
        avg_validation_precision = sum(float(item.get("precision", 0.0)) for item in validation_per_class.values()) / max(len(validation_per_class), 1)
        avg_test_recall = sum(float(item.get("recall", 0.0)) for item in test_per_class.values()) / max(len(test_per_class), 1)
        avg_test_precision = sum(float(item.get("precision", 0.0)) for item in test_per_class.values()) / max(len(test_per_class), 1)
        model_comparison.append({
            "name": result["name"],
            "model_type": result["model_type"],
            "validation_accuracy": validation["accuracy"],
            "validation_f1": validation["macro_f1"],
            "validation_weighted_f1": validation["weighted_f1"],
            "validation_roc_auc_ovo": validation["roc_auc_ovo"],
            "validation_recall": avg_validation_recall,
            "validation_precision": avg_validation_precision,
            "validation_confusion_matrix": validation["confusion_matrix"],
            "validation_per_class": validation_per_class,
            "test_accuracy": test["accuracy"],
            "test_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"],
            "test_roc_auc_ovo": test["roc_auc_ovo"],
            "test_recall": avg_test_recall,
            "test_precision": avg_test_precision,
            "test_confusion_matrix": test["confusion_matrix"],
            "test_per_class": test_per_class,
            "calibration_status": "Confidence tracked in prediction log; formal calibration curve not included in the saved metadata.",
        })
        if experiment["name"].startswith("Model A") and (
            best_result is None or result["validation"]["macro_f1"] > best_result["validation"]["macro_f1"]
        ):
            best_result = result
            best_pipeline = pipeline

    baseline_pred = train_df["target"].mode().iloc[0]
    baseline_accuracy = (test_df["target"] == baseline_pred).mean()
    grouped_results = grouped_evaluation(best_pipeline, pd.concat([train_df, validation_df, test_df], axis=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

    model_metadata = {
        "model_name": best_result["name"],
        "model_type": best_result["model_type"],
        "training_date": datetime.utcnow().isoformat() + "Z",
        "dataset_version": str(PROCESSED_DATA.name),
        "input_columns": best_result["feature_columns"],
        "target_labels": {k: v["original_label"] for k, v in RISK_MAPPING.items()},
        "training_sample_size": len(train_df),
        "validation_results": best_result["validation"],
        "test_results": best_result["test"],
        "grouped_evaluation": grouped_results,
        "model_comparison": model_comparison,
        "known_limitations": [
            "Text fields such as complaint and imaging may leak diagnostic labels.",
            "Model B is an experimental baseline using only pre-diagnostic structured features.",
            "This prototype is investigational and not clinically validated.",
        ],
        "model_version": f"v1.{datetime.utcnow():%Y%m%d%H%M%S}",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

    save_report(results, best_result, grouped_results, baseline_accuracy)
    save_error_analysis(test_df, best_pipeline)


if __name__ == "__main__":
    train_best_model()