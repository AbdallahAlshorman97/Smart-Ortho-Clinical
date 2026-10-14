from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

RISK_MAPPING = {
    0: {
        "level": "Level 1 – Highest urgency",
        "original_label": "Urgent",
        "recommended_time": "Within 6 hours",
    },
    1: {
        "level": "Level 2 – Intermediate urgency",
        "original_label": "Emergency",
        "recommended_time": "Within 12–24 hours",
    },
    2: {
        "level": "Level 3 – Elective",
        "original_label": "Chronic",
        "recommended_time": "Elective appointment",
    },
}

LAB_PATTERNS = {
    "WBC": r"WBC\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    "CRP": r"CRP\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    "ESR": r"ESR\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    "Hemoglobin": r"(?:Hemoglobin|Hb)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
}

MISSING_LAB_VALUES = {"—", "-", "", "�", None}

NUMERIC_VALIDATION_RULES = {
    "Age": (0, 110),
    "HR": (30, 220),
    "Temp": (30.0, 43.0),
    "VAS": (0, 10),
    "ODI": (0, 100),
    "Systolic_BP": (50, 250),
    "Diastolic_BP": (30, 150),
}

SEX_MAPPING = {
    "M": "Male",
    "F": "Female",
    "m": "Male",
    "f": "Female",
}

INPUT_COLUMNS = [
    "ID",
    "Age",
    "Sex",
    "Complaint",
    "HR",
    "BP",
    "Temp",
    "Labs",
    "Imaging (X-rays/MRI)",
    "VAS",
    "ODI (%)",
    "Risk",
]

@dataclass
class CleanedLabResult:
    WBC: Optional[float] = None
    CRP: Optional[float] = None
    ESR: Optional[float] = None
    Hemoglobin: Optional[float] = None
    Other_Lab_Text: Optional[str] = None
    Labs_Available: int = 0


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Clinical Dataset")


def parse_bp(value: object) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    if text in {"", "NA", "N/A", "nan"}:
        return None, None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", text)
    if match:
        try:
            return int(float(match.group(1))), int(float(match.group(2)))
        except ValueError:
            return None, None
    return None, None


def parse_labs(value: object) -> CleanedLabResult:
    if pd.isna(value):
        return CleanedLabResult()
    text = str(value).strip()
    if text in MISSING_LAB_VALUES:
        return CleanedLabResult()

    remaining = text
    parsed = {}
    for field, pattern in LAB_PATTERNS.items():
        match = re.search(pattern, remaining, flags=re.IGNORECASE)
        if match:
            try:
                parsed[field] = float(match.group(1))
            except ValueError:
                parsed[field] = None
            remaining = re.sub(pattern, "", remaining, flags=re.IGNORECASE)

    # Remove separators and stray punctuation
    remaining = re.sub(r"[,:;]+", " ", remaining).strip()
    remaining = re.sub(r"\s{2,}", " ", remaining).strip()
    if remaining == "":
        remaining = None

    return CleanedLabResult(
        WBC=parsed.get("WBC"),
        CRP=parsed.get("CRP"),
        ESR=parsed.get("ESR"),
        Hemoglobin=parsed.get("Hemoglobin"),
        Other_Lab_Text=remaining,
        Labs_Available=1,
    )


def standardize_sex(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return SEX_MAPPING.get(text, "Unknown")


def map_risk_label(value: object) -> Dict[str, object]:
    if pd.isna(value):
        raise ValueError("Risk label is missing")
    text = str(value)
    if "Urgent" in text:
        code = 0
    elif "Emergency" in text:
        code = 1
    elif "Chronic" in text:
        code = 2
    else:
        raise ValueError(f"Unrecognized risk label: {value}")
    mapping = RISK_MAPPING[code]
    return {
        "dataset_label": mapping["original_label"],
        "target": code,
        "risk_level": mapping["level"],
        "recommended_time": mapping["recommended_time"],
    }


def validate_numeric(value: Optional[float], key: str) -> bool:
    if value is None or pd.isna(value):
        return True
    low, high = NUMERIC_VALIDATION_RULES[key]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return low <= numeric <= high


def build_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.rename(columns={"Imaging (X-rays/MRI)": "Imaging", "ODI (%)": "ODI"})
    cleaned["Sex"] = cleaned["Sex"].apply(standardize_sex)
    cleaned[["Systolic_BP", "Diastolic_BP"]] = cleaned["BP"].apply(
        lambda x: pd.Series(parse_bp(x), index=["Systolic_BP", "Diastolic_BP"])
    )

    labs = cleaned["Labs"].apply(parse_labs)
    cleaned["WBC"] = labs.apply(lambda r: r.WBC)
    cleaned["CRP"] = labs.apply(lambda r: r.CRP)
    cleaned["ESR"] = labs.apply(lambda r: r.ESR)
    cleaned["Hemoglobin"] = labs.apply(lambda r: r.Hemoglobin)
    cleaned["Other_Lab_Text"] = labs.apply(lambda r: r.Other_Lab_Text)
    cleaned["Labs_Available"] = labs.apply(lambda r: r.Labs_Available)

    risk_info = cleaned["Risk"].apply(map_risk_label).apply(pd.Series)
    cleaned = pd.concat([cleaned, risk_info], axis=1)

    for field in ["Age", "HR", "Temp", "VAS", "ODI", "Systolic_BP", "Diastolic_BP"]:
        cleaned[f"{field}_valid"] = cleaned[field].apply(lambda x: validate_numeric(x, field))
    cleaned["Sex_valid"] = cleaned["Sex"].isin({"Male", "Female"})
    cleaned["Review_Required"] = ~(
        cleaned[["Age_valid", "HR_valid", "Temp_valid", "VAS_valid", "ODI_valid", "Systolic_BP_valid", "Diastolic_BP_valid", "Sex_valid"]].all(axis=1)
    )

    return cleaned


def save_processed_files(root: Path) -> None:
    original_path = Path("Smart Ortho Clinical Dataset.xlsx")
    output_original = root / "data" / "original" / "01_original_dataset.xlsx"
    output_processed = root / "data" / "processed" / "02_cleaned_dataset.csv"
    output_model_ready = root / "data" / "processed" / "03_model_ready_dataset.csv"
    output_rules = root / "models" / "risk_label_mapping.json"

    output_original.parent.mkdir(parents=True, exist_ok=True)
    output_processed.parent.mkdir(parents=True, exist_ok=True)
    output_model_ready.parent.mkdir(parents=True, exist_ok=True)
    output_rules.parent.mkdir(parents=True, exist_ok=True)

    if not output_original.exists():
        original_path.replace(output_original)

    df = load_dataset(output_original)
    cleaned = build_clean_dataframe(df)

    cleaned.to_csv(output_processed, index=False)

    model_ready_columns = [
        "ID",
        "Age",
        "Sex",
        "Complaint",
        "HR",
        "Systolic_BP",
        "Diastolic_BP",
        "Temp",
        "Labs",
        "WBC",
        "CRP",
        "ESR",
        "Hemoglobin",
        "Other_Lab_Text",
        "Labs_Available",
        "Imaging",
        "VAS",
        "ODI",
        "dataset_label",
        "target",
        "risk_level",
        "recommended_time",
        "Age_valid",
        "HR_valid",
        "Temp_valid",
        "VAS_valid",
        "ODI_valid",
        "Systolic_BP_valid",
        "Diastolic_BP_valid",
        "Sex_valid",
        "Review_Required",
    ]
    cleaned[model_ready_columns].to_csv(output_model_ready, index=False)

    with open(output_rules, "w", encoding="utf-8") as handle:
        json.dump(RISK_MAPPING, handle, ensure_ascii=False, indent=2)

    print(f"Saved cleaned dataset to {output_processed}")
    print(f"Saved model-ready dataset to {output_model_ready}")
    print(f"Saved risk label mapping to {output_rules}")


if __name__ == "__main__":
    save_processed_files(Path(__file__).resolve().parent.parent)