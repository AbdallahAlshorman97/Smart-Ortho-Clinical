from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_gui_uses_current_risk_labels():
    text = (ROOT / "smart_ortho_gui.py").read_text(encoding="utf-8")
    assert "High risk" in text
    assert "Moderate risk" in text
    assert "Low risk" in text
    assert "Model probability" in text
    assert "prediction_logs_v2.csv" in text
