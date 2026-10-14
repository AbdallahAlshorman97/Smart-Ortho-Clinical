from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_case

HISTORY_FILE = ROOT / "models" / "prediction_logs_v2.csv"
HISTORY_LIMIT = 10

SEVERITY_COLORS = {
    0: "#d32f2f",
    1: "#fb8c00",
    2: "#fbc02d",
}

LEGEND_ITEMS = [
    ("High risk", "#d32f2f", "Red / severe orange"),
    ("Moderate risk", "#fb8c00", "Remaining orange"),
    ("Low risk", "#fbc02d", "Yellow / green"),
]

SEX_OPTIONS = ["M", "F"]
VAS_OPTIONS = [str(i) for i in range(11)]
ODI_OPTIONS = [str(i) for i in range(0, 101, 5)]

COMMON_COMPLAINTS = [
    "Chronic mechanical low back pain",
    "Cauda equina syndrome",
    "Pelvic fracture, haemodynamically unstable",
    "Acute spinal cord compression",
    "Rheumatoid arthritis, hands",
    "Acute lumbar radiculopathy with motor weakness",
    "Lumbar disc prolapse, no red flags",
    "Chronic cervical spondylosis",
    "Spinal epidural abscess with cord compression",
    "Open tibial fracture (Right)",
    "Closed distal radius fracture (Right)",
    "Closed clavicle fracture (Right)",
    "Severe diabetic foot infection with sepsis (Right)",
]

COMMON_IMAGING = [
    "Clinical diagnosis, no imaging required",
    "X-ray: open tibial shaft fracture, Gustilo IIIA",
    "X-ray: mild degenerative changes, no red-flag features",
    "MRI: mild disc desiccation, no nerve compression",
    "X-ray: knee OA, joint space narrowing, osteophytes",
    "X-ray: open comminuted tibial fracture with bone exposure",
    "CT: vertically unstable pelvic ring fracture with active bleeding",
    "X-ray: closed distal radius fracture, dorsal angulation",
    "X-ray: hip OA with subchondral sclerosis and osteophytes",
    "X-ray: closed clavicle fracture with moderate displacement",
    "MRI: osteomyelitis with abscess formation, forefoot",
    "CT angiogram: arterial occlusion with distal hypoperfusion",
]

INPUT_FIELDS = [
    ("Age", "age", "0-110", "spin"),
    ("Sex", "sex", "Select sex", "combo"),
    ("Complaint", "complaint", "Select or type complaint", "combo_text"),
    ("Heart rate", "hr", "30-220", "spin"),
    ("Systolic BP", "systolic_bp", "50-250", "spin"),
    ("Diastolic BP", "diastolic_bp", "30-150", "spin"),
    ("Temperature", "temperature", "30.0-43.0", "spin_float"),
    ("Laboratory results", "labs", "WBC 12.2, CRP 130", "text"),
    ("Imaging report", "imaging", "Select or type imaging summary", "combo_text"),
    ("VAS", "vas", "0-10", "combo"),
    ("ODI", "odi", "0-100", "combo"),
]

DEFAULT_INPUT = {
    "age": 30,
    "sex": "F",
    "complaint": "Chronic mechanical low back pain",
    "hr": 72,
    "systolic_bp": 118,
    "diastolic_bp": 76,
    "temperature": 36.7,
    "labs": "",
    "imaging": "X-ray: mild degenerative changes, no red-flag features",
    "vas": 3,
    "odi": 12,
}

EXAMPLE_INPUTS = {
    "High risk": {
        "age": 54,
        "sex": "M",
        "complaint": "Open tibial fracture with severe pain",
        "hr": 108,
        "systolic_bp": 140,
        "diastolic_bp": 88,
        "temperature": 37.8,
        "labs": "WBC 15.2, CRP 130",
        "imaging": "X-ray: open comminuted tibial fracture",
        "vas": 10,
        "odi": 90,
    },
    "Moderate risk": {
        "age": 41,
        "sex": "F",
        "complaint": "Acute wrist injury after fall",
        "hr": 96,
        "systolic_bp": 130,
        "diastolic_bp": 82,
        "temperature": 36.8,
        "labs": "",
        "imaging": "X-ray: closed distal radius fracture without displacement",
        "vas": 7,
        "odi": 55,
    },
    "Low risk": {
        "age": 30,
        "sex": "F",
        "complaint": "Chronic mechanical low back pain",
        "hr": 72,
        "systolic_bp": 118,
        "diastolic_bp": 76,
        "temperature": 36.7,
        "labs": "",
        "imaging": "X-ray: mild degenerative changes, no acute abnormality",
        "vas": 3,
        "odi": 12,
    },
}

RANGE_HINTS = {
    "age": (0, 110),
    "hr": (30, 220),
    "systolic_bp": (50, 250),
    "diastolic_bp": (30, 150),
    "temperature": (30.0, 43.0),
    "vas": (0, 10),
    "odi": (0, 100),
}


class SmartOrthoGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Smart-Ortho Prediction Tool")
        self.geometry("900x820")
        self.resizable(True, True)
        self._widgets: dict[str, tk.Widget] = {}
        self._configure_style()
        self._build_ui()
        self._update_history()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=(None, 20, "bold"), foreground="#1f3a66")
        style.configure("Section.TLabelframe", background="#f4f8fc", borderwidth=1, relief="solid")
        style.configure("Section.TLabelframe.Label", font=(None, 12, "bold"), foreground="#1f3a66")
        style.configure("Action.TButton", font=(None, 11, "bold"), padding=8)
        style.configure("Result.TLabel", font=(None, 12, "bold"))
        style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")

    def _load_evaluation_metrics(self) -> dict[str, object]:
        metadata_path = ROOT / "models" / "model_metadata.json"
        if not metadata_path.exists():
            return {
                "model_name": "Unavailable",
                "model_type": "Unavailable",
                "validation_results": {"accuracy": None, "macro_f1": None, "weighted_f1": None, "roc_auc_ovo": None, "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "per_class": {}},
                "test_results": {"accuracy": None, "macro_f1": None, "weighted_f1": None, "roc_auc_ovo": None, "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "per_class": {}},
            }
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_model_comparison(self) -> list[dict[str, float | str]]:
        metadata = self._load_evaluation_metrics()
        comparison = metadata.get("model_comparison")
        if isinstance(comparison, list) and comparison:
            return comparison

        report_path = ROOT / "reports" / "model_results.md"
        if not report_path.exists():
            return []

        models: list[dict[str, float | str]] = []
        current: dict[str, float | str] = {}
        for raw_line in report_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("### "):
                if current:
                    models.append(current)
                current = {"model": line[4:].strip()}
                continue
            if not current or ":" not in line:
                continue
            if line.startswith("- Validation accuracy:"):
                current["validation_accuracy"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("- Validation macro F1:"):
                current["validation_f1"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("- Test accuracy:"):
                current["test_accuracy"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("- Test macro F1:"):
                current["test_f1"] = float(line.split(":", 1)[1].strip())
        if current:
            models.append(current)
        return models

    def _build_ui(self) -> None:
        canvas = tk.Canvas(self, borderwidth=0, background="#e9eef4")
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        root_frame = ttk.Frame(canvas, padding=14)
        canvas.create_window((0, 0), window=root_frame, anchor="nw")

        def on_frame_configure(event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        root_frame.bind("<Configure>", on_frame_configure)

        header = ttk.Label(root_frame, text="Smart-Ortho Prediction Tool", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=3, sticky="w")

        subtitle = ttk.Label(
            root_frame,
            text="A local clinical decision-support prototype for urgency categorization.",
            wraplength=780,
            justify="left",
        )
        subtitle.grid(row=1, column=0, columnspan=3, pady=(4, 12), sticky="w")

        notebook = ttk.Notebook(root_frame)
        notebook.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self._notebook = notebook

        input_tab = ttk.Frame(notebook, padding=8)
        self._result_tab = ttk.Frame(notebook, padding=8)
        history_tab = ttk.Frame(notebook, padding=8)
        evaluation_tab = ttk.Frame(notebook, padding=8)
        about_tab = ttk.Frame(notebook, padding=8)
        notebook.add(input_tab, text="Inputs")
        notebook.add(self._result_tab, text="Results")
        notebook.add(history_tab, text="History")
        notebook.add(evaluation_tab, text="Evaluation")
        notebook.add(about_tab, text="About")

        entry_frame = ttk.LabelFrame(input_tab, text="Patient and Clinical Inputs", style="Section.TLabelframe", padding=12)
        entry_frame.grid(row=0, column=0, sticky="nsew")

        for idx, (label_text, field_name, hint, field_type) in enumerate(INPUT_FIELDS):
            label = ttk.Label(entry_frame, text=label_text)
            label.grid(row=idx, column=0, sticky="w", pady=8, padx=(4, 8))

            if field_type == "combo":
                if field_name == "sex":
                    entry = ttk.Combobox(entry_frame, values=SEX_OPTIONS, state="readonly", width=22)
                    entry.set(SEX_OPTIONS[0])
                elif field_name == "vas":
                    entry = ttk.Combobox(entry_frame, values=VAS_OPTIONS, state="readonly", width=22)
                    entry.set(VAS_OPTIONS[-1])
                elif field_name == "odi":
                    entry = ttk.Combobox(entry_frame, values=ODI_OPTIONS, state="readonly", width=22)
                    entry.set(ODI_OPTIONS[-1])
                else:
                    entry = ttk.Combobox(entry_frame, values=[], width=22)
            elif field_type == "combo_text":
                values = COMMON_COMPLAINTS if field_name == "complaint" else COMMON_IMAGING
                entry = ttk.Combobox(entry_frame, values=values, width=60)
                entry.set(values[0])
            elif field_type == "spin":
                entry = ttk.Spinbox(entry_frame, from_=RANGE_HINTS[field_name][0], to=RANGE_HINTS[field_name][1], width=24)
                entry.set(str(DEFAULT_INPUT[field_name]))
            elif field_type == "spin_float":
                entry = ttk.Spinbox(entry_frame, from_=RANGE_HINTS[field_name][0], to=RANGE_HINTS[field_name][1], increment=0.1, width=24)
                entry.set(str(DEFAULT_INPUT[field_name]))
            else:
                entry = tk.Text(entry_frame, height=3, width=60, wrap="word")
                entry.insert("1.0", hint if field_name != "imaging" else "")

            self._widgets[field_name] = entry
            if isinstance(entry, tk.Text):
                entry.grid(row=idx, column=1, sticky="we", pady=6, padx=(0, 4), columnspan=2)
            else:
                entry.grid(row=idx, column=1, sticky="w", pady=6, padx=(0, 4))

            hint_label = ttk.Label(entry_frame, text=f"Hint: {hint}", foreground="#555555")
            hint_label.grid(row=idx, column=2, sticky="e", padx=(8, 4))

        actions_frame = ttk.Frame(input_tab)
        actions_frame.grid(row=1, column=0, pady=(10, 8), sticky="we")
        actions_frame.columnconfigure(0, weight=1)

        predict_button = ttk.Button(actions_frame, text="Predict urgency", style="Action.TButton", command=self.on_predict)
        predict_button.grid(row=0, column=0, sticky="w", padx=(0, 6))

        clear_result_button = ttk.Button(actions_frame, text="Clear results", style="Action.TButton", command=self._clear_results)
        clear_result_button.grid(row=0, column=1, sticky="w", padx=(0, 6))

        example_label = ttk.Label(actions_frame, text="Quick example buttons:", font=(None, 10, "bold"))
        example_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 4))

        high_risk_button = ttk.Button(actions_frame, text="High risk example", style="Action.TButton", command=lambda: self._load_example("High risk"))
        high_risk_button.grid(row=2, column=0, sticky="w", padx=(0, 6))

        moderate_risk_button = ttk.Button(actions_frame, text="Moderate risk example", style="Action.TButton", command=lambda: self._load_example("Moderate risk"))
        moderate_risk_button.grid(row=2, column=1, sticky="w", padx=(0, 6))

        low_risk_button = ttk.Button(actions_frame, text="Low risk example", style="Action.TButton", command=lambda: self._load_example("Low risk"))
        low_risk_button.grid(row=2, column=2, sticky="w")

        self._latest_summary_frame = ttk.LabelFrame(input_tab, text="Latest prediction summary", style="Section.TLabelframe", padding=12)
        self._latest_summary_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        self._latest_summary_label = ttk.Label(self._latest_summary_frame, text="No prediction yet.", foreground="#333333", wraplength=760, justify="left")
        self._latest_summary_label.pack(fill="both", expand=True)

        self._result_frame = ttk.LabelFrame(self._result_tab, text="Prediction output", style="Section.TLabelframe", padding=12)
        self._result_frame.grid(row=0, column=0, sticky="nsew")

        self._level_label = tk.Label(self._result_frame, text="Awaiting prediction...", font=(None, 16, "bold"), background="#ffffff", anchor="w", padx=10, pady=10)
        self._level_label.grid(row=0, column=0, sticky="we", pady=(0, 10))

        clear_results_button = ttk.Button(self._result_frame, text="Clear results", style="Action.TButton", command=self._clear_results)
        clear_results_button.grid(row=0, column=1, sticky="e", padx=(10, 0), pady=(0, 10))

        self._result_table = ttk.Treeview(self._result_frame, columns=("field", "value"), show="headings", height=5)
        self._result_table.heading("field", text="Metric")
        self._result_table.heading("value", text="Result")
        self._result_table.column("field", width=200, anchor="w")
        self._result_table.column("value", width=560, anchor="w")
        self._result_table.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self._input_summary_frame = ttk.LabelFrame(self._result_frame, text="Input summary", padding=10)
        self._input_summary_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._input_summary_table = ttk.Treeview(self._input_summary_frame, columns=("field", "value"), show="headings", height=12)
        self._input_summary_table.heading("field", text="Input")
        self._input_summary_table.heading("value", text="Value")
        self._input_summary_table.column("field", width=200, anchor="w")
        self._input_summary_table.column("value", width=560, anchor="w")
        self._input_summary_table.pack(side="left", fill="both", expand=True)

        input_summary_scroll = ttk.Scrollbar(self._input_summary_frame, orient="vertical", command=self._input_summary_table.yview)
        input_summary_scroll.pack(side="right", fill="y")
        self._input_summary_table.configure(yscrollcommand=input_summary_scroll.set)

        self._result_plot_frame = ttk.LabelFrame(self._result_frame, text="Prediction metrics", padding=10)
        self._result_plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._result_figure = Figure(figsize=(6, 2.5), dpi=100)
        self._result_axes = self._result_figure.add_subplot(111)
        self._result_canvas = FigureCanvasTkAgg(self._result_figure, master=self._result_plot_frame)
        self._result_canvas.draw()
        self._result_canvas.get_tk_widget().pack(fill="both", expand=True)

        legend_container = ttk.Frame(self._result_frame)
        legend_container.grid(row=4, column=0, columnspan=2, sticky="we", pady=(10, 0))
        for idx in range(6):
            legend_container.columnconfigure(idx, weight=1)
        for index, (label_text, color, note) in enumerate(LEGEND_ITEMS):
            block = tk.Label(legend_container, background=color, width=2, height=1)
            block.grid(row=0, column=index * 2, sticky="w", padx=(0, 4))
            ttk.Label(legend_container, text=f"{label_text}: {note}", foreground="#333333").grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 18))

        self._result_details_text = tk.Text(self._result_frame, height=4, wrap="word", state="disabled", background="#eef4fb", bd=0, padx=10, pady=10)
        self._result_details_text.grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 0))

        self._result_frame.columnconfigure(1, weight=1)

        history_actions = ttk.Frame(history_tab)
        history_actions.grid(row=0, column=0, sticky="we", pady=(0, 8))
        history_actions.columnconfigure(0, weight=1)

        clear_button = ttk.Button(history_actions, text="Clear history", style="Action.TButton", command=self._clear_history)
        clear_button.grid(row=0, column=0, sticky="w")

        self._history_summary_label = ttk.Label(history_actions, text="Last prediction: none", foreground="#333333")
        self._history_summary_label.grid(row=0, column=1, sticky="e")

        history_panel = ttk.Frame(history_tab)
        history_panel.grid(row=1, column=0, sticky="nsew")
        history_panel.columnconfigure(0, weight=2)
        history_panel.columnconfigure(1, weight=1)

        self._history_frame = ttk.LabelFrame(history_panel, text="Prediction history", style="Section.TLabelframe", padding=12)
        self._history_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._history_view = ttk.Treeview(self._history_frame, columns=("timestamp", "level", "label", "confidence", "review"), show="headings", height=8)
        self._history_view.heading("timestamp", text="Timestamp")
        self._history_view.heading("level", text="Risk level")
        self._history_view.heading("label", text="Dataset label")
        self._history_view.heading("confidence", text="Model probability")
        self._history_view.heading("review", text="Clinician review")
        self._history_view.column("timestamp", width=160, anchor="w")
        self._history_view.column("level", width=180, anchor="w")
        self._history_view.column("label", width=120, anchor="w")
        self._history_view.column("confidence", width=90, anchor="center")
        self._history_view.column("review", width=120, anchor="center")
        self._history_view.grid(row=0, column=0, sticky="nsew")

        history_scroll = ttk.Scrollbar(self._history_frame, orient="vertical", command=self._history_view.yview)
        history_scroll.grid(row=0, column=1, sticky="ns")
        self._history_view.configure(yscrollcommand=history_scroll.set)

        self._history_frame.columnconfigure(0, weight=1)

        self._history_plot_frame = ttk.LabelFrame(history_panel, text="History distribution", style="Section.TLabelframe", padding=12)
        self._history_plot_frame.grid(row=0, column=1, sticky="nsew")
        self._history_plot_frame.columnconfigure(0, weight=1)
        self._history_plot_frame.rowconfigure(0, weight=1)

        self._history_figure = Figure(figsize=(4, 3), dpi=100)
        self._history_axes = self._history_figure.add_subplot(111)
        self._history_canvas = FigureCanvasTkAgg(self._history_figure, master=self._history_plot_frame)
        self._history_canvas.draw()
        self._history_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        disclaimer = ttk.Label(input_tab, text="This prototype is investigational and not a substitute for clinical judgment.", foreground="#7d7d7d", wraplength=780, justify="left")
        disclaimer.grid(row=2, column=0, sticky="w", pady=(8, 0))

        evaluation_data = self._load_evaluation_metrics()
        validation = evaluation_data.get("validation_results", {})
        test = evaluation_data.get("test_results", {})
        class_metrics = validation.get("per_class", {})

        metric_frame = ttk.LabelFrame(evaluation_tab, text="Model performance summary", style="Section.TLabelframe", padding=12)
        metric_frame.grid(row=0, column=0, sticky="nsew")
        metrics_table = ttk.Treeview(metric_frame, columns=("metric", "value"), show="headings", height=10)
        metrics_table.heading("metric", text="Metric")
        metrics_table.heading("value", text="Value")
        metrics_table.column("metric", width=220, anchor="w")
        metrics_table.column("value", width=220, anchor="w")
        metrics_table.grid(row=0, column=0, sticky="nsew")
        metric_rows = [
            ("Model", evaluation_data.get("model_name", "Unknown")),
            ("Model type", evaluation_data.get("model_type", "Unknown")),
            ("Validation accuracy", f"{validation.get('accuracy'):.3f}" if validation.get("accuracy") is not None else "N/A"),
            ("Validation F1-score", f"{validation.get('macro_f1'):.3f}" if validation.get("macro_f1") is not None else "N/A"),
            ("Validation weighted F1", f"{validation.get('weighted_f1'):.3f}" if validation.get("weighted_f1") is not None else "N/A"),
            ("Validation ROC-AUC", f"{validation.get('roc_auc_ovo'):.3f}" if validation.get("roc_auc_ovo") is not None else "N/A"),
            ("Test accuracy", f"{test.get('accuracy'):.3f}" if test.get("accuracy") is not None else "N/A"),
            ("Test F1-score", f"{test.get('macro_f1'):.3f}" if test.get("macro_f1") is not None else "N/A"),
            ("Sensitivity / recall", f"{sum(float(v.get('recall', 0.0)) for v in class_metrics.values())/max(len(class_metrics), 1):.3f}" if class_metrics else "N/A"),
            ("Calibration status", "Confidence tracked in prediction log; formal calibration curve not included in the saved metadata."),
        ]
        for metric, value in metric_rows:
            metrics_table.insert("", "end", values=(metric, value))

        comparison_rows = self._load_model_comparison()
        comparison_frame = ttk.LabelFrame(evaluation_tab, text="Model comparison", style="Section.TLabelframe", padding=12)
        comparison_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        comparison_table = ttk.Treeview(
            comparison_frame,
            columns=("model", "validation_acc", "validation_f1", "validation_recall", "test_acc", "test_f1", "test_recall", "calibration"),
            show="headings",
            height=10,
        )
        comparison_table.heading("model", text="Model")
        comparison_table.heading("validation_acc", text="Val acc")
        comparison_table.heading("validation_f1", text="Val F1")
        comparison_table.heading("validation_recall", text="Val recall")
        comparison_table.heading("test_acc", text="Test acc")
        comparison_table.heading("test_f1", text="Test F1")
        comparison_table.heading("test_recall", text="Test recall")
        comparison_table.heading("calibration", text="Calibration")
        comparison_table.column("model", width=240, anchor="w")
        for col in ["validation_acc", "validation_f1", "validation_recall", "test_acc", "test_f1", "test_recall"]:
            comparison_table.column(col, width=110, anchor="center")
        comparison_table.column("calibration", width=260, anchor="w")
        if comparison_rows:
            for row in comparison_rows:
                if "validation_accuracy" in row:
                    val_acc = float(row.get("validation_accuracy", 0.0))
                    val_f1 = float(row.get("validation_f1", 0.0))
                    val_recall = float(row.get("validation_recall", row.get("validation_f1", 0.0)))
                    test_acc = float(row.get("test_accuracy", 0.0))
                    test_f1 = float(row.get("test_f1", 0.0))
                    test_recall = float(row.get("test_recall", row.get("test_f1", 0.0)))
                    calibration = "Confidence tracked in prediction log; formal calibration curve not included in the saved metadata."
                    model_name = row.get("model", "Unknown")
                else:
                    val_acc = float(row.get("validation_accuracy", row.get("validation", {}).get("accuracy", 0.0)))
                    val_f1 = float(row.get("validation_f1", row.get("validation", {}).get("macro_f1", 0.0)))
                    val_recall = float(row.get("validation_recall", 0.0))
                    test_acc = float(row.get("test_accuracy", row.get("test", {}).get("accuracy", 0.0)))
                    test_f1 = float(row.get("test_f1", row.get("test", {}).get("macro_f1", 0.0)))
                    test_recall = float(row.get("test_recall", 0.0))
                    calibration = row.get("calibration_status", "Confidence tracked in prediction log; formal calibration curve not included in the saved metadata.")
                    model_name = row.get("name", row.get("model", "Unknown"))

                comparison_table.insert(
                    "",
                    "end",
                    values=(
                        model_name,
                        f"{val_acc:.4f}",
                        f"{val_f1:.4f}",
                        f"{val_recall:.4f}",
                        f"{test_acc:.4f}",
                        f"{test_f1:.4f}",
                        f"{test_recall:.4f}",
                        calibration,
                    ),
                )
        comparison_table.grid(row=0, column=0, sticky="nsew")

        comparison_plot_frame = ttk.LabelFrame(evaluation_tab, text="Test F1 comparison", style="Section.TLabelframe", padding=12)
        comparison_plot_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        comparison_figure = Figure(figsize=(7, 3), dpi=100)
        comparison_axes = comparison_figure.add_subplot(111)
        if comparison_rows:
            labels = []
            values = []
            for row in comparison_rows:
                labels.append(row.get("model", row.get("name", "Unknown")))
                if "test_f1" in row:
                    values.append(float(row.get("test_f1", 0.0)))
                elif isinstance(row.get("test", {}), dict):
                    values.append(float(row["test"].get("macro_f1", 0.0)))
            comparison_axes.bar(labels[:len(values)], values, color="#4a90e2")
            comparison_axes.set_ylabel("Test macro F1")
            comparison_axes.set_title("Model comparison by F1-score")
            comparison_axes.tick_params(axis="x", rotation=20)
            comparison_axes.set_ylim(0, 1.05)
        else:
            comparison_axes.text(0.5, 0.5, "No comparison data available", ha="center", va="center")
            comparison_axes.axis("off")
        comparison_canvas = FigureCanvasTkAgg(comparison_figure, master=comparison_plot_frame)
        comparison_canvas.draw()
        comparison_canvas.get_tk_widget().pack(fill="both", expand=True)

        for idx, row in enumerate(comparison_rows[:4], start=3):
            model_name = row.get("model", row.get("name", "Unknown"))
            validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            test = row.get("test") if isinstance(row.get("test"), dict) else {}
            validation_metrics = validation.get("per_class", row.get("validation_per_class", {}))
            test_metrics = test.get("per_class", row.get("test_per_class", {}))

            model_frame = ttk.LabelFrame(evaluation_tab, text=f"{model_name} details", style="Section.TLabelframe", padding=12)
            model_frame.grid(row=idx, column=0, sticky="nsew", pady=(10, 0))

            summary_table = ttk.Treeview(model_frame, columns=("metric", "value"), show="headings", height=8)
            summary_table.heading("metric", text="Metric")
            summary_table.heading("value", text="Value")
            summary_table.column("metric", width=260, anchor="w")
            summary_table.column("value", width=220, anchor="w")
            summary_table.grid(row=0, column=0, sticky="nsew")

            display_validation = validation if validation else row
            display_test = test if test else {}
            val_acc = float(display_validation.get("accuracy", row.get("validation_accuracy", 0.0)))
            val_f1 = float(display_validation.get("macro_f1", row.get("validation_f1", 0.0)))
            val_recall = float(row.get("validation_recall", sum(float(v.get("recall", 0.0)) for v in validation_metrics.values()) / max(len(validation_metrics), 1)))
            test_acc = float(display_test.get("accuracy", row.get("test_accuracy", 0.0)))
            test_f1 = float(display_test.get("macro_f1", row.get("test_f1", 0.0)))
            test_recall = float(row.get("test_recall", sum(float(v.get("recall", 0.0)) for v in test_metrics.values()) / max(len(test_metrics), 1)))
            calibration = row.get("calibration_status", "Confidence tracked in prediction log; formal calibration curve not included in the saved metadata.")
            summary_rows = [
                ("Validated accuracy", f"{val_acc:.4f}"),
                ("F1-score", f"{val_f1:.4f}"),
                ("Sensitivity / recall", f"{val_recall:.4f}"),
                ("Test accuracy", f"{test_acc:.4f}"),
                ("Test F1-score", f"{test_f1:.4f}"),
                ("Test recall", f"{test_recall:.4f}"),
                ("Calibration note", calibration),
            ]
            for metric, value in summary_rows:
                summary_table.insert("", "end", values=(metric, value))

            class_table = ttk.Treeview(model_frame, columns=("class", "recall", "precision", "f1"), show="headings", height=5)
            class_table.heading("class", text="Class")
            class_table.heading("recall", text="Recall")
            class_table.heading("precision", text="Precision")
            class_table.heading("f1", text="F1-score")
            class_table.column("class", width=140, anchor="w")
            for key, values in validation_metrics.items():
                label_name = {"0": "High risk", "1": "Moderate risk", "2": "Low risk"}.get(str(key), str(key))
                class_table.insert("", "end", values=(
                    label_name,
                    f"{float(values.get('recall', 0.0)):.3f}",
                    f"{float(values.get('precision', 0.0)):.3f}",
                    f"{float(values.get('f1', 0.0)):.3f}",
                ))
            class_table.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

            cm_values = validation.get("confusion_matrix", row.get("validation_confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
            cm_frame = ttk.LabelFrame(model_frame, text="Validation confusion matrix", style="Section.TLabelframe", padding=12)
            cm_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
            cm_figure = Figure(figsize=(4.5, 3.8), dpi=100)
            cm_axes = cm_figure.add_subplot(111)
            cm_array = cm_values
            cm_axes.imshow(cm_array, cmap="Blues")
            cm_axes.set_xticks([0, 1, 2])
            cm_axes.set_yticks([0, 1, 2])
            cm_axes.set_xticklabels(["High risk", "Moderate risk", "Low risk"])
            cm_axes.set_yticklabels(["High risk", "Moderate risk", "Low risk"])
            for i in range(len(cm_array)):
                for j in range(len(cm_array[0])):
                    cm_axes.text(j, i, str(cm_array[i][j]), ha="center", va="center", color="black")
            cm_canvas = FigureCanvasTkAgg(cm_figure, master=cm_frame)
            cm_canvas.draw()
            cm_canvas.get_tk_widget().pack(fill="both", expand=True)

        about_frame = ttk.LabelFrame(about_tab, text="About this tool", style="Section.TLabelframe", padding=12)
        about_frame.grid(row=0, column=0, sticky="nsew")
        about_text = (
            "Smart-Ortho Prediction Tool\n"
            "Created by Abdallah Al-Shorman\n"
            "Email: aalshorm@emich.edu\n"
            "Organization: EMU / CyberRiget\n\n"
            "This tool is a local investigational prototype for urgency categorization. "
            "It is not intended for clinical decision-making or diagnosis."
        )
        about_label = ttk.Label(about_frame, text=about_text, wraplength=780, justify="left", foreground="#333333")
        about_label.grid(row=0, column=0, sticky="nw")

    def _load_example(self, label: str) -> None:
        example_data = EXAMPLE_INPUTS.get(label, DEFAULT_INPUT)
        for field_name, value in example_data.items():
            widget = self._widgets[field_name]
            if isinstance(widget, tk.Text):
                widget.delete("1.0", "end")
                widget.insert("1.0", str(value))
            elif hasattr(widget, "set"):
                widget.set(str(value))
            else:
                widget.delete(0, "end")
                widget.insert(0, str(value))

    def on_predict(self) -> None:
        try:
            input_data = self._collect_inputs()
            result = predict_case(input_data)
            self._display_result(result)
            self._notebook.select(self._result_tab)
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))

    def _collect_inputs(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for _, field_name, _, field_type in INPUT_FIELDS:
            widget = self._widgets[field_name]
            if isinstance(widget, tk.Text):
                value = widget.get("1.0", "end").strip()
            else:
                value = widget.get().strip()
            if field_name in {"age", "hr", "systolic_bp", "diastolic_bp", "vas", "odi"}:
                if value == "":
                    raise ValueError(f"{field_name.replace('_', ' ').title()} is required")
                try:
                    numeric = int(float(value))
                except ValueError:
                    raise ValueError(f"{field_name.replace('_', ' ').title()} must be a number")
                low, high = RANGE_HINTS[field_name]
                if not (low <= numeric <= high):
                    raise ValueError(f"{field_name.replace('_', ' ').title()} must be between {low} and {high}")
                values[field_name] = numeric
            elif field_name == "temperature":
                if value == "":
                    raise ValueError("Temperature is required")
                try:
                    numeric = float(value)
                except ValueError:
                    raise ValueError("Temperature must be a number")
                low, high = RANGE_HINTS[field_name]
                if not (low <= numeric <= high):
                    raise ValueError(f"Temperature must be between {low} and {high}")
                values[field_name] = numeric
            elif field_name == "sex":
                if value not in SEX_OPTIONS:
                    raise ValueError("Sex must be M or F")
                values[field_name] = value
            else:
                if field_name in {"complaint", "imaging"} and value == "":
                    raise ValueError(f"{field_name.replace('_', ' ').title()} is required")
                values[field_name] = value
        return values

    def _display_result(self, result: dict[str, object]) -> None:
        severity = result.get("internal_level", "Unknown")
        severity_color = SEVERITY_COLORS.get(result.get("predicted_target"), "#546e7a")
        self._level_label.config(text=severity, background=severity_color, foreground="#ffffff")

        for row in self._result_table.get_children():
            self._result_table.delete(row)

        rows = [
            ("Prediction source", result.get("prediction_source", "Model prediction")),
            ("Model prediction", result.get("model_prediction") or result.get("dataset_label")),
            ("Final prediction", result.get("final_prediction") or result.get("dataset_label")),
            ("Model probability", f"{float(result.get('model_probability', result.get('confidence', 0.0))) * 100:.1f}%"),
            ("Rule triggered", result.get("rule_trigger") or result.get("rule_triggered") or "None"),
            ("Recommended assessment", result.get("recommended_time")),
            ("Requires clinician review", "Yes" if result.get("requires_clinician_review") else "No"),
        ]

        for field, value in rows:
            self._result_table.insert("", "end", values=(field, value))

        details = [
            f"Prediction ID: {result.get('prediction_id')}",
        ]
        if result.get("review_reasons"):
            reasons = ", ".join(result.get('review_reasons', []))
            details.insert(0, f"Review reasons: {reasons}")

        self._result_details_text.config(state="normal")
        self._result_details_text.delete("1.0", "end")
        self._result_details_text.insert("1.0", "\n".join(details))
        self._result_details_text.config(state="disabled")

        self._input_summary_table.delete(*self._input_summary_table.get_children())
        for field_name, value in result.get("input_values", {}).items():
            self._input_summary_table.insert("", "end", values=(field_name.replace("_", " ").title(), value))

        self._refresh_result_plot(result)
        self._refresh_latest_summary(result)
        self._update_history()

    def _clear_results(self) -> None:
        self._level_label.config(text="Awaiting prediction...", background="#ffffff", foreground="#000000")
        for row in self._result_table.get_children():
            self._result_table.delete(row)
        self._result_details_text.config(state="normal")
        self._result_details_text.delete("1.0", "end")
        self._result_details_text.config(state="disabled")
        self._input_summary_table.delete(*self._input_summary_table.get_children())
        self._result_axes.clear()
        self._result_axes.set_title("Current prediction metrics")
        self._result_canvas.draw()
        if hasattr(self, '_latest_summary_label'):
            self._latest_summary_label.config(text="No prediction yet.")

    def _refresh_result_plot(self, result: dict[str, object]) -> None:
        model_probability = float(result.get('model_probability', result.get('confidence', 0.0))) * 100
        review_required = 1.0 if result.get('requires_clinician_review') else 0.0
        colors = ['#3f51b5', '#d32f2f']
        self._result_axes.clear()
        bars = self._result_axes.bar(['Model probability', 'Review required'], [model_probability, review_required * 100], color=colors)
        self._result_axes.set_ylim(0, 110)
        self._result_axes.set_ylabel('Percent')
        self._result_axes.set_title('Current prediction metrics')
        self._result_axes.set_xticks([0, 1])
        self._result_axes.set_xticklabels(['Model probability', 'Review required'], rotation=0)
        self._result_axes.bar_label(bars, labels=[f"{model_probability:.1f}%", 'Yes' if review_required else 'No'], padding=3)
        self._result_canvas.draw()

    def _refresh_latest_summary(self, result: dict[str, object]) -> None:
        summary_text = (
            f"Last prediction: {result.get('internal_level')} ({result.get('dataset_label')})\n"
            f"Model probability: {float(result.get('model_probability', result.get('confidence', 0.0))) * 100:.1f}%\n"
            f"Review required: {'Yes' if result.get('requires_clinician_review') else 'No'}"
        )
        if hasattr(self, '_latest_summary_label'):
            self._latest_summary_label.config(text=summary_text)
        if hasattr(self, '_history_summary_label'):
            self._history_summary_label.config(text=f"Last prediction: {result.get('internal_level')} ({result.get('dataset_label')})")

    def _clear_history(self) -> None:
        if messagebox.askyesno("Clear history", "Are you sure you want to clear all saved prediction history?"):
            try:
                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()
                self._update_history()
            except Exception as exc:
                messagebox.showerror("Clear history failed", str(exc))

    def _refresh_history_plot(self, history_entries: list[tuple[str, str, str, str, str]]) -> None:
        counts = {"High risk": 0, "Moderate risk": 0, "Low risk": 0}
        for _, internal_level, dataset_label, _, _ in history_entries:
            if dataset_label in counts:
                counts[dataset_label] += 1
            elif "High risk" in internal_level:
                counts["High risk"] += 1
            elif "Moderate risk" in internal_level:
                counts["Moderate risk"] += 1
            elif "Low risk" in internal_level:
                counts["Low risk"] += 1
        labels = list(counts.keys())
        values = list(counts.values())
        self._history_axes.clear()
        self._history_axes.bar(labels, values, color=["#d32f2f", "#fb8c00", "#fbc02d"])
        self._history_axes.set_title("Recent risk distribution")
        self._history_axes.set_ylabel("Count")
        self._history_axes.set_ylim(0, max(values + [1]))
        for i, value in enumerate(values):
            self._history_axes.text(i, value + 0.05, str(value), ha="center", va="bottom")
        self._history_canvas.draw()

    def _load_history(self) -> list[tuple[str, str, str, str, str]]:
        if not HISTORY_FILE.exists():
            return []
        entries: list[tuple[str, str, str, str, str]] = []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    confidence_text = "0.0%"
                    try:
                        confidence_text = f"{float(row.get('confidence', '0')) * 100:.1f}%"
                    except (TypeError, ValueError):
                        pass
                    entries.append(
                        (
                            row.get("timestamp", ""),
                            row.get("internal_level", ""),
                            row.get("dataset_label", ""),
                            confidence_text,
                            "Yes" if str(row.get("requires_clinician_review", "False")).strip().lower() in {"true", "1", "yes"} else "No",
                        )
                    )
        except Exception:
            return []
        return entries[-HISTORY_LIMIT:][::-1]

    def _update_history(self) -> None:
        history_entries = self._load_history()
        self._history_view.delete(*self._history_view.get_children())
        if history_entries:
            for entry in history_entries:
                self._history_view.insert("", "end", values=entry)
            latest = history_entries[0]
            self._history_summary_label.config(text=f"Last prediction: {latest[0]} – {latest[1]} ({latest[2]})")
        else:
            self._history_view.insert("", "end", values=("No history available", "", "", "", ""))
            self._history_summary_label.config(text="Last prediction: none")
        self._refresh_history_plot(history_entries)


def main() -> None:
    app = SmartOrthoGUI()
    app.mainloop()


if __name__ == "__main__":
    main()