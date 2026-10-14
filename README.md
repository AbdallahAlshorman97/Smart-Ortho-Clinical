# Smart-Ortho Clinical

**Smart-Ortho** is an investigational, clinician-facing Python desktop prototype for classifying synthetic musculoskeletal presentations into three relative urgency categories: **High risk**, **Moderate risk**, and **Low risk**.

> **Research software only.** Not clinically validated. Not for patient-facing triage, diagnosis, autonomous decisions, or clinical deployment.

## Features
- Structured clinical inputs, including age, sex, vital signs, pain and disability scores
- Free-text complaint, laboratory findings, and imaging description
- Machine-learning predictions with a separate rule-based high-risk override
- Original model probability displayed separately from the rule decision
- Clinician-review flags, local prediction history, and evaluation visualizations
- Example cases for exploring the interface

## Research context
The study uses a **synthetic 15,000-case dataset** (5,000 cases per risk class). Reported evaluation performance is internal to that synthetic data and **must not be interpreted as real-world clinical accuracy or safety**. The rule layer has known limitations with negation, spelling variation, and multilingual inputs. A patient-facing navigator is a future concept and is **not included** in this implementation.

## Repository structure
- `smart_ortho_gui.py` — desktop interface
- `src/` — prediction, feature engineering, data preparation, training, evaluation, and rule engine
- `models/clinical_rules.json` — investigational keyword rules
- `models/risk_label_mapping.json` — source-to-display risk mappings
- `results/` — research figures where available
- `tests/` — regression tests (as added)

The trained pipeline and any data required for execution may need to be obtained separately. Do not commit patient data, private logs, secrets, or credentials.

## Running locally
Install Python 3.10+ and the dependencies from `requirements.txt`, then run:
```bash
python smart_ortho_gui.py
```
The application needs a compatible trained model file in `models/` for predictions. Without that artifact, the interface cannot perform inference.

## Development history
This repository was created after the project's earlier local development. Historical milestones should be documented from dated source files, records, or authentic prior commits. **The GitHub commit timestamps record actual uploads and subsequent changes, not retroactive development activity.**

## License
No license has been granted yet; contact the author before reuse.

## Research notice
Outputs require clinician review and should never determine care. Formal external clinical validation, probability calibration, rule-layer validation, and local clinical governance would be necessary before clinical use.
