from __future__ import annotations

import csv
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.prepare_data import (
    RISK_MAPPING,
    map_risk_label,
    parse_bp,
    parse_labs,
    standardize_sex,
)
from src.rules_engine import check_high_risk_rules

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "smart_ortho_pipeline.joblib"
MAPPING_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_label_mapping.json"
METADATA_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"
LOG_PATH = Path(__file__).resolve().parent.parent / "models" / "prediction_logs.csv"

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def load_pipeline(path: Optional[Path] = None) -> Pipeline:
    pipeline_path = path or MODEL_PATH
    return joblib.load(pipeline_path)


def load_risk_mapping(path: Optional[Path] = None) -> Dict[int, Dict[str, str]]:
    mapping_path = path or MAPPING_PATH
    with open(mapping_path, "r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)
    return {int(k): v for k, v in raw_mapping.items()}


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_input_row(data: Dict[str, Any]) -> Dict[str, Any]:
    systolic, diastolic = parse_bp(f"{data.get('systolic_bp', '')}/{data.get('diastolic_bp', '')}")
    if systolic is None or diastolic is None:
        # Try parsing separate fields or keep as None
        systolic = safe_float(data.get("systolic_bp"))
        diastolic = safe_float(data.get("diastolic_bp"))
    labs = data.get("labs") or ""
    if labs.strip() in {"—", "-"}:
        labs = ""
    lab_result = parse_labs(labs)

    input_row = {
        "ID": data.get("patient_id") or data.get("id") or "",
        "Age": safe_float(data.get("age")),
        "Sex": standardize_sex(data.get("sex")),
        "Complaint": data.get("complaint") or "",
        "HR": safe_float(data.get("hr")),
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Temp": safe_float(data.get("temperature")),
        "Labs": labs,
        "WBC": lab_result.WBC,
        "CRP": lab_result.CRP,
        "ESR": lab_result.ESR,
        "Hemoglobin": lab_result.Hemoglobin,
        "Other_Lab_Text": lab_result.Other_Lab_Text,
        "Labs_Available": lab_result.Labs_Available,
        "Imaging": data.get("imaging") or "",
        "VAS": safe_float(data.get("vas")),
        "ODI": safe_float(data.get("odi")),
    }
    return input_row


def predict_case(input_data: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = load_pipeline()
    mapping = load_risk_mapping()
    row = build_input_row(input_data)
    features = {
        "Age": row["Age"],
        "Sex": row["Sex"],
        "Complaint": row["Complaint"],
        "HR": row["HR"],
        "Systolic_BP": row["Systolic_BP"],
        "Diastolic_BP": row["Diastolic_BP"],
        "Temp": row["Temp"],
        "Labs": row["Labs"],
        "WBC": row["WBC"],
        "CRP": row["CRP"],
        "ESR": row["ESR"],
        "Hemoglobin": row["Hemoglobin"],
        "Labs_Available": row["Labs_Available"],
        "Imaging": row["Imaging"],
        "VAS": row["VAS"],
        "ODI": row["ODI"],
    }
    feature_df = pd.DataFrame([features])
    predicted = pipeline.predict(feature_df)[0]
    probabilities = pipeline.predict_proba(feature_df)[0]
    confidence = float(round(float(np.max(probabilities)), 4))

    rule = check_high_risk_rules(row["Complaint"], row["Imaging"])
    rule_triggered = None
    prediction_source = "Model prediction"
    if rule is not None:
        rule_triggered = rule.get("description")
        predicted = 0
        confidence = max(confidence, 0.9)
        prediction_source = "Rule override"

    risk_info = mapping.get(int(predicted), {})
    requires_clinician_review = False
    reasons: list[str] = []
    if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
        requires_clinician_review = True
        reasons.append("Low model confidence")
    if rule is not None:
        requires_clinician_review = True
        if prediction_source == "Rule override":
            reasons.append("Rule-based urgent override")
        if predicted != 0:
            reasons.append("Rule conflict with model prediction")
    if not row["Complaint"] or not row["Imaging"]:
        requires_clinician_review = True
        reasons.append("Missing complaint or imaging text")

    prediction_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    result = {
        "prediction_id": prediction_id,
        "patient_id": row["ID"],
        "input_values": features,
        "model_version": load_model_version(),
        "predicted_target": int(predicted),
        "internal_level": risk_info.get("level"),
        "dataset_label": risk_info.get("original_label"),
        "recommended_time": risk_info.get("recommended_time"),
        "confidence": confidence,
        "prediction_source": prediction_source,
        "rule_triggered": rule_triggered,
        "requires_clinician_review": requires_clinician_review,
        "review_reasons": reasons,
        "timestamp": timestamp,
    }
    _append_prediction_log(result)
    return result


def load_model_version() -> str:
    if not METADATA_PATH.exists():
        return "unknown"
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata.get("model_version", "unknown")


def _append_prediction_log(entry: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_id",
        "patient_id",
        "model_version",
        "predicted_target",
        "internal_level",
        "dataset_label",
        "recommended_time",
        "confidence",
        "prediction_source",
        "rule_triggered",
        "requires_clinician_review",
        "review_reasons",
        "timestamp",
        "input_values",
    ]
    file_exists = LOG_PATH.exists()
    write_fieldnames = fieldnames
    if file_exists:
        with open(LOG_PATH, "r", encoding="utf-8", newline="") as existing_handle:
            existing_header = next(csv.reader(existing_handle), [])
        if "prediction_source" not in existing_header:
            write_fieldnames = [name for name in fieldnames if name != "prediction_source"]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            **{k: entry.get(k) for k in writer.fieldnames if k != "input_values"},
            "input_values": json.dumps(entry.get("input_values", {}), ensure_ascii=False),
        })