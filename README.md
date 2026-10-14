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

## Six-Month Project History

Smart-Ortho was developed locally before this GitHub repository was published. The timeline below documents the project history across the six-month period leading up to the public repository. GitHub commit timestamps remain the actual upload dates and have not been backdated.

| Period | Project history |
| --- | --- |
| **April 2026** | Project planning and early concept-development period. No dated repository artifact is currently available to verify a specific implementation milestone for this month. |
| **May 2026** | Early local development period. No dated repository artifact is currently available to verify a specific implementation milestone for this month. |
| **June 2026** | Continued local development period. No dated repository artifact is currently available to verify a specific implementation milestone for this month. |
| **July 2026** | Pre-release development period. No dated repository artifact is currently available to verify a specific implementation milestone for this month. |
| **August 2026** | Model-development activity is directly documented by saved metadata, including a recorded model-training run on **August 12, 2026**. The project included synthetic-data preprocessing, classifier development, model comparison, and internal evaluation. |
| **September 2026** | The project advanced through structured-feature comparisons, ablation analysis, information-leakage investigation, safety-layer review, and manuscript refinement. |
| **October 2026** | Reviewer-driven revisions aligned the implementation with the manuscript: High/Moderate/Low risk terminology, clearer separation of model probability from rule overrides, GUI refinement, result figures, regression checks, and GitHub publication. |
| **October 14, 2026** | The public GitHub repository was initialized and the Smart-Ortho codebase and documentation were uploaded. |

> **Timeline note:** Months without a dated preserved project artifact are shown explicitly as unverified local-development periods rather than being assigned invented milestones. As older dated files are recovered, this table can be updated with additional verified details.

## License
No license has been granted yet; contact the author before reuse.

## Research notice
Outputs require clinician review and should never determine care. Formal external clinical validation, probability calibration, rule-layer validation, and local clinical governance would be necessary before clinical use.
