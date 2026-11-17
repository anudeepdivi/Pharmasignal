"""PharmaSignal interactive dashboard.

Run from the project root:
    .venv/bin/streamlit run src/pharmasignal/dashboard/app.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pharmasignal.dashboard.logic import (
    curves,
    drift_table,
    explain_record,
    parse_upload,
    read_jsonl,
    score_rows,
    threshold_metrics,
)
from pharmasignal.data.schema import FIELDS_BY_NAME, PREDICTORS
from pharmasignal.features.fusion import assemble_features
from pharmasignal.models.artifact import load_artifact

MODEL_DIR = Path(os.environ.get("PHARMASIGNAL_MODEL_DIR", "artifacts/models/m4"))
DATA_DIR = Path(os.environ.get("PHARMASIGNAL_DATA_DIR", "artifacts/data/seed42"))
MONITORING_DIR = Path(os.environ.get("PHARMASIGNAL_MONITORING_DIR", "artifacts/monitoring"))
SIMULATED_DIR = Path(os.environ.get("PHARMASIGNAL_SIMULATED_DIR", "artifacts/data/simulated"))

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#94a3b8"

st.set_page_config(page_title="PharmaSignal", page_icon="💊", layout="wide")


@st.cache_resource
def get_model():
    return load_artifact(MODEL_DIR)


@st.cache_data
def get_split(name: str) -> list[dict]:
    return read_jsonl(DATA_DIR / f"{name}.jsonl")


@st.cache_data
def get_report() -> dict:
    return json.loads((MODEL_DIR / "report.json").read_text())


@st.cache_data
def get_test_scores() -> tuple[np.ndarray, np.ndarray]:
    model, _ = get_model()
    records = get_split("test")
    scores = model.predict_proba(assemble_features(records))[:, 1]
    return np.array([r["signal"] for r in records]), scores


def missing_artifacts() -> list[str]:
    needed = [MODEL_DIR / "metadata.json", DATA_DIR / "test.jsonl"]
    return [str(path) for path in needed if not path.exists()]


def gauge(score: float, threshold: float) -> go.Figure:
    flagged = score >= threshold
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ORANGE if flagged else BLUE},
                "threshold": {"line": {"color": RED, "width": 3}, "value": threshold * 100},
            },
            title={"text": "Review-priority score"},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def confusion_figure(matrix: list[list[int]]) -> go.Figure:
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{labels[i][j]}<br><b>{matrix[i][j]}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted: routine", "Predicted: review"],
            y=["Actual: routine", "Actual: review"],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_autorange="reversed")
    return fig


# --------------------------------------------------------------------------- header
st.title("💊 PharmaSignal — adverse-event report triage")
st.caption(
    "Machine-learning demo that ranks adverse-event reports for human review using structured "
    "fields **and** free-text narratives. All data is synthetic; this is not a clinical tool."
)

absent = missing_artifacts()
if absent:
    st.error(
        "Model or data artifacts not found: "
        + ", ".join(absent)
        + ". Run `make artifacts` (see README → Quick start) and reload."
    )
    st.stop()

model, metadata = get_model()
report = get_report()
threshold = float(metadata["threshold"])
test_metrics = report["test_at_selected_threshold"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("ROC-AUC (held-out test)", f"{test_metrics['roc_auc']:.3f}")
k2.metric("Recall at operating point", f"{test_metrics['recall']:.0%}")
k3.metric("Precision at operating point", f"{test_metrics['precision']:.0%}")
k4.metric("Model features", "1,321", help="Structured + rule-based NLP + TF-IDF text features")

tab_score, tab_batch, tab_perf, tab_drift, tab_about = st.tabs(
    ["🔎 Score a report", "📄 Batch scoring", "📈 Model performance", "🌊 Data drift", "ℹ️ About"]
)

# --------------------------------------------------------------------------- single report
with tab_score:
    examples = get_split("test")[:40]
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Report details")
        pick = st.selectbox(
            "Load an example report",
            range(len(examples)),
            format_func=lambda i: (
                f"Example {i + 1} — {examples[i]['treatment']}, "
                f"age {examples[i]['age'] if examples[i]['age'] is not None else 'n/a'}"
            ),
        )
        ex = examples[pick]
        with st.form("single"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 18, 90, value=ex["age"], placeholder="unknown")
            sex_opts = [None, *FIELDS_BY_NAME["sex"].choices]
            sex = c2.selectbox(
                "Sex", sex_opts, index=sex_opts.index(ex["sex"]), format_func=lambda v: v or "—"
            )
            treatment = c3.selectbox(
                "Treatment",
                FIELDS_BY_NAME["treatment"].choices,
                index=FIELDS_BY_NAME["treatment"].choices.index(ex["treatment"]),
            )
            c4, c5 = st.columns(2)
            dose = c4.number_input("Dose (mg)", 0.1, 500.0, value=ex["dose_mg"], placeholder="—")
            days = c5.number_input("Exposure (days)", 1, 365, value=ex["exposure_days"])
            c6, c7 = st.columns(2)
            egfr = c6.number_input("Baseline eGFR", 10.0, 150.0, value=ex["baseline_egfr"])
            meds = c7.number_input(
                "Concomitant meds", 0, 15, value=ex["concomitant_med_count"], placeholder="—"
            )
            narrative = st.text_area("Narrative", ex["narrative"], height=130, max_chars=4000)
            submitted = st.form_submit_button("Score report", type="primary")

    record = {
        "age": None if age is None else int(age),
        "sex": sex,
        "treatment": treatment,
        "dose_mg": None if dose is None else float(dose),
        "exposure_days": None if days is None else int(days),
        "baseline_egfr": None if egfr is None else float(egfr),
        "concomitant_med_count": None if meds is None else int(meds),
        "narrative": narrative,
    }
    with right:
        st.subheader("Result")
        if not narrative.strip():
            st.warning("Enter a narrative to score this report.")
        else:
            score = float(model.predict_proba(assemble_features([record]))[0, 1])
            st.plotly_chart(gauge(score, threshold), width="stretch")
            if score >= threshold:
                st.error(f"**Flag for review** — score is above the {threshold:.1%} threshold.")
            else:
                st.success(f"**Routine** — score is below the {threshold:.1%} threshold.")
            explanation = explain_record(model, record).iloc[::-1]
            fig = go.Figure(
                go.Bar(
                    x=explanation["contribution"],
                    y=explanation["label"],
                    orientation="h",
                    marker_color=[ORANGE if v > 0 else BLUE for v in explanation["contribution"]],
                )
            )
            fig.update_layout(
                title="Why? Top drivers of this score",
                xaxis_title="← lowers priority · raises priority →",
                height=330,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Each bar compares this report with an average training report. "
                "It explains the model's score, not medical causation."
            )

# --------------------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Score many reports at once")
    st.write(
        "Upload a CSV with one report per row. Required columns: "
        + ", ".join(f"`{c}`" for c in PREDICTORS)
        + ". Optional fields may be left blank. Extra columns (e.g. your own ID) are kept."
    )
    template = pd.DataFrame(get_split("test")[:25])[["report_id", *PREDICTORS]]
    for name in PREDICTORS:
        if FIELDS_BY_NAME[name].kind == "integer":
            template[name] = template[name].astype("Int64")
    st.download_button(
        "⬇️ Download a sample CSV (25 reports)",
        template.to_csv(index=False),
        file_name="pharmasignal_sample.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is not None:
        try:
            frame = pd.read_csv(upload, dtype=object, keep_default_na=False, na_values=[""])
            valid, errors = parse_upload(frame)
        except Exception as exc:  # noqa: BLE001 - shown to the user, not logged
            st.error(f"Could not read file: {exc}")
        else:
            if not errors.empty:
                st.warning(f"{len(errors)} row(s) skipped because of invalid values.")
                st.dataframe(errors, hide_index=True, width="stretch")
            if valid:
                scored = score_rows(model, threshold, valid)
                out = frame.reset_index(drop=True).assign(row=range(1, len(frame) + 1))
                out = out.merge(scored, on="row").drop(columns="row")
                out = out.sort_values("priority_score", ascending=False)
                lead = ["flag_for_review", "priority_score"]
                out = out[lead + [c for c in out.columns if c not in lead]]
                n_flag = int((out["flag_for_review"] == "YES").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Reports scored", len(out))
                m2.metric("Flagged for review", n_flag)
                m3.metric("Share flagged", f"{n_flag / len(out):.0%}")
                st.dataframe(
                    out,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "priority_score": st.column_config.ProgressColumn(
                            "Priority score", min_value=0.0, max_value=1.0, format="%.3f"
                        )
                    },
                )
                buffer = io.StringIO()
                out.to_csv(buffer, index=False)
                st.download_button(
                    "⬇️ Download scored CSV",
                    buffer.getvalue(),
                    file_name="pharmasignal_scored.csv",
                    mime="text/csv",
                    type="primary",
                )

# --------------------------------------------------------------------------- performance
with tab_perf:
    labels, scores = get_test_scores()
    st.subheader("How well does the model rank reports?")
    st.caption("Evaluated once on 600 held-out synthetic reports that were never used in training.")
    chosen = st.slider(
        "Decision threshold — drag to see the precision / recall trade-off",
        0.01,
        0.99,
        value=threshold,
        step=0.01,
        format="%.3f",
        help="Starts at the operating threshold chosen on validation data (recall target 80%).",
    )
    m = threshold_metrics(labels, scores, chosen)
    a, b, c, d = st.columns(4)
    a.metric("Recall (caught review cases)", f"{m['recall']:.0%}")
    b.metric("Precision (flags that were right)", f"{m['precision']:.0%}")
    c.metric("F1 score", f"{m['f1']:.2f}")
    d.metric("Reports flagged", f"{m['flagged_share']:.0%}")

    curve = curves(labels, scores)
    op = threshold_metrics(labels, scores, chosen)
    fpr_op = op["confusion_matrix"][0][1] / max(1, sum(op["confusion_matrix"][0]))
    g1, g2, g3 = st.columns(3)
    roc = go.Figure()
    roc.add_scatter(x=curve["fpr"], y=curve["tpr"], name="Model", line_color=BLUE)
    roc.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash"))
    roc.add_scatter(
        x=[fpr_op], y=[op["recall"]], mode="markers", name="Threshold", marker_color=RED
    )
    roc.update_layout(
        title=f"ROC curve (AUC {test_metrics['roc_auc']:.3f})",
        xaxis_title="False-positive rate",
        yaxis_title="True-positive rate",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    g1.plotly_chart(roc, width="stretch")
    pr = go.Figure()
    pr.add_scatter(x=curve["recall"], y=curve["precision"], name="Model", line_color=BLUE)
    pr.add_hline(y=float(np.mean(labels)), line=dict(color=GREY, dash="dash"))
    pr.add_scatter(
        x=[op["recall"]], y=[op["precision"]], mode="markers", name="Threshold", marker_color=RED
    )
    pr.update_layout(
        title=f"Precision–recall (AP {test_metrics['average_precision']:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    g2.plotly_chart(pr, width="stretch")
    g3.plotly_chart(confusion_figure(m["confusion_matrix"]), width="stretch")

    st.subheader("Does combining text + structured data help?")
    comparison = report["validation_comparison_at_0_5"]
    names = {"structured": "Structured fields only", "text": "Narrative text only"}
    names["fused"] = "Combined (selected)"
    ap = [comparison[v]["average_precision"] for v in ("structured", "text", "fused")]
    bar = go.Figure(
        go.Bar(
            x=list(names.values()),
            y=ap,
            text=[f"{v:.3f}" for v in ap],
            textposition="outside",
            marker_color=[GREY, GREY, BLUE],
        )
    )
    bar.update_layout(
        yaxis_title="Validation average precision",
        yaxis_range=[0, max(ap) * 1.25],
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(bar, width="stretch")

# --------------------------------------------------------------------------- drift
with tab_drift:
    st.subheader("Is incoming data still like the training data?")
    st.caption(
        "Compares a new batch of reports against the training data, field by field. "
        "Batches here are simulated: one unchanged, one with deliberately shifted inputs."
    )
    scenario = st.radio(
        "Batch", ["shifted", "unchanged"], horizontal=True, format_func=str.capitalize
    )
    report_path = MONITORING_DIR / scenario / "report.json"
    if not report_path.exists():
        st.info(f"No drift report at `{report_path}`. Run `make artifacts` to generate it.")
    else:
        drift = json.loads(report_path.read_text())
        summary = drift["summary"]
        if summary["overall_status"] == "drift_detected":
            st.error(f"⚠️ Drift detected — {summary['alerts']} check(s) raised an alert.")
        else:
            st.success("✅ No drift detected — all checks within thresholds.")
        table = drift_table(drift)
        left, right = st.columns([1.2, 1])
        with left:
            st.dataframe(
                table.style.map(
                    lambda s: f"color: {RED}; font-weight: 600" if s == "alert" else "",
                    subset=["status"],
                ),
                hide_index=True,
                width="stretch",
                height=420,
                column_config={"value": st.column_config.NumberColumn(format="%.3f")},
            )
        with right:
            field = st.selectbox(
                "Compare distributions",
                ["dose_mg", "baseline_egfr", "exposure_days", "concomitant_med_count", "age"],
            )
            batch_path = SIMULATED_DIR / scenario / "batch.jsonl"
            if batch_path.exists():
                ref = [r[field] for r in get_split("train") if r[field] is not None]
                new = [r[field] for r in read_jsonl(batch_path) if r[field] is not None]
                hist = go.Figure()
                hist.add_histogram(
                    x=ref, name="Training data", histnorm="probability", marker_color=BLUE
                )
                hist.add_histogram(
                    x=new, name="New batch", histnorm="probability", marker_color=ORANGE
                )
                hist.update_traces(opacity=0.6)
                hist.update_layout(
                    barmode="overlay",
                    height=380,
                    xaxis_title=field,
                    yaxis_title="Share of reports",
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(hist, width="stretch")

# --------------------------------------------------------------------------- about
with tab_about:
    st.subheader("What this project demonstrates")
    st.markdown(
        """
- **Data engineering** — synthetic data generator, strict schema validation, leakage-safe
  train/validation/test splits.
- **NLP + tabular ML** — spaCy tokenisation, negation-aware rule features and TF-IDF fused
  with structured fields in a scikit-learn pipeline (logistic regression).
- **Responsible evaluation** — model and threshold chosen on validation only; the test set is
  scored exactly once.
- **MLOps** — MLflow experiment tracking, a versioned and checksummed model artifact,
  a FastAPI prediction service, Docker packaging and data-drift monitoring.
- **Quality** — 200+ automated tests, linting and fully reproducible runs.

**Disclaimer:** every report is fictional and generated by code. Scores reflect the synthetic
generator, not real patient risk. Not for clinical use.
"""
    )
    st.caption(
        f"Model version `{metadata['model_version']}` · schema `{metadata['schema_version']}`"
    )
