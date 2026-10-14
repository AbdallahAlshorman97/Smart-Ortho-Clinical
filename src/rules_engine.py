import json
import re
from pathlib import Path
from typing import Dict, List, Optional

RULES_FILE = Path(__file__).resolve().parent.parent / "models" / "clinical_rules.json"


def load_clinical_rules(path: Optional[Path] = None) -> List[Dict[str, str]]:
    rules_path = path or RULES_FILE
    with open(rules_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_high_risk_rules(complaint: str, imaging: str) -> Optional[Dict[str, str]]:
    complaint_text = (complaint or "").lower()
    imaging_text = (imaging or "").lower()
    rules = load_clinical_rules()
    for rule in rules:
        pattern = rule["pattern"]
        if re.search(pattern, complaint_text) or re.search(pattern, imaging_text):
            return rule
    return None