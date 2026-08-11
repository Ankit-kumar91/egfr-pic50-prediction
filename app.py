"""Streamlit UI for EGFR pIC50 prediction.

Run with:
    streamlit run app.py

Predicts against a local PredictionPipeline by default; check "Use API
backend" in the sidebar to instead call a running FastAPI server (see
api/main.py) at API_URL.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from src.pipeline.predict_pipeline import (  # noqa: E402
    MODEL_DESCRIPTIONS,
    MODEL_LABELS,
    PredictionPipeline,
)

API_URL = os.getenv("API_URL")

EXAMPLE_MOLECULES = {
    "-- Select --": "",
    "Gefitinib (approved EGFR inhibitor)": (
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    ),
    "Erlotinib (approved EGFR inhibitor)": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "Afatinib (approved EGFR inhibitor)": (
        "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1"
    ),
    "Aspirin (not a kinase inhibitor -- triggers AD warning)": (
        "CC(=O)Oc1ccccc1C(=O)O"
    ),
}

st.set_page_config(
    page_title="EGFR pIC50 Prediction",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors mirror .streamlit/config.toml's theme -- this block only adds
# structure Streamlit's theme system doesn't cover (the header banner,
# model description cards), it doesn't redefine the palette.
st.markdown(
    """
    <style>
    :root {
        /* This app's theme is locked to light via .streamlit/config.toml,
        with no in-app toggle. Without this, some browsers (notably
        Chrome's "force dark" on Android) heuristically invert the page
        anyway, breaking contrast on custom-styled elements below. */
        color-scheme: light;
    }
    .app-header {
        padding: 1.5rem 1.75rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #146C94 0%, #0C4A6E 100%);
        color: #FFFFFF;
        margin-bottom: 1.25rem;
    }
    .app-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 1.9rem;
    }
    .app-header p {
        color: #DCEEF6;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .model-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 0.5rem;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.6rem;
        background-color: var(--secondary-background-color);
    }
    .model-card .model-name {
        font-weight: 600;
        color: var(--text-color);
        font-size: 0.92rem;
    }
    .model-card .model-tagline {
        font-size: 0.8rem;
        color: #146C94;
        margin-bottom: 0.25rem;
    }
    .model-card .model-detail {
        font-size: 0.78rem;
        color: var(--text-color);
        opacity: 0.75;
        line-height: 1.35;
    }
    .model-card .model-rmse {
        display: inline-block;
        margin-top: 0.35rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #146C94;
        background-color: rgba(20, 108, 148, 0.15);
        border-radius: 0.3rem;
        padding: 0.1rem 0.4rem;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 0.5rem;
        padding: 0.75rem 0.9rem 0.5rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_local_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


def draw_molecule(smiles: str, size=(280, 280)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def predict_via_api(smiles_list: list[str], models: list[str]) -> list[dict]:
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"smiles_list": smiles_list, "models": models},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["results"]


def predict_locally(smiles_list: list[str], models: list[str]) -> list[dict]:
    pipeline = get_local_pipeline()
    return pipeline.predict_batch(smiles_list, models)


def run_predictions(
    smiles_list: list[str], models: list[str], use_api: bool
) -> list[dict]:
    if use_api:
        return predict_via_api(smiles_list, models)
    return predict_locally(smiles_list, models)


def render_comparison_chart(result: dict, models: list[str]) -> None:
    """Point estimate + uncertainty range per model, one hue (position on the
    y-axis already carries model identity, so color doesn't need to)."""
    rows = [
        {
            "model": MODEL_LABELS[name],
            "pic50": result["predictions"][name]["pic50"],
            "lower": result["predictions"][name]["lower"],
            "upper": result["predictions"][name]["upper"],
        }
        for name in models
    ]
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(y=alt.Y("model:N", sort=list(df["model"]), title=None))
    x_scale = alt.Scale(zero=False)

    range_layer = base.mark_rule(color="#146C94", strokeWidth=2).encode(
        x=alt.X("lower:Q", title="pIC50", scale=x_scale),
        x2="upper:Q",
    )
    point_layer = base.mark_circle(size=130, color="#0C4A6E").encode(
        x=alt.X("pic50:Q", scale=x_scale),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("pic50:Q", title="pIC50", format=".2f"),
            alt.Tooltip("lower:Q", title="Lower", format=".2f"),
            alt.Tooltip("upper:Q", title="Upper", format=".2f"),
        ],
    )
    text_layer = base.mark_text(
        align="left", dx=14, fontSize=12, color="#1A2332"
    ).encode(
        x=alt.X("pic50:Q", scale=x_scale),
        text=alt.Text("pic50:Q", format=".2f"),
    )

    chart = (
        (range_layer + point_layer + text_layer)
        .properties(height=38 * len(models) + 10)
        .configure_axis(gridColor="#E5EBEF", domainColor="#D7E3EA", tickColor="#D7E3EA")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def render_prediction_cards(result: dict, models: list[str]) -> None:
    ad = result["applicability_domain"]
    if not ad["in_domain"]:
        st.warning(
            f"Outside the model's applicability domain: the nearest training "
            f"compound has only {ad['nearest_neighbor_similarity']:.2f} Tanimoto "
            f"similarity (threshold {ad['threshold']:.2f}). Treat this "
            f"prediction as a rough estimate, not a hard rejection.",
            icon="⚠️",
        )
    else:
        st.success(
            f"Within the applicability domain "
            f"(nearest training compound: {ad['nearest_neighbor_similarity']:.2f} "
            f"Tanimoto similarity).",
            icon="✅",
        )

    cols = st.columns(len(models))
    for col, name in zip(cols, models):
        pred = result["predictions"][name]
        with col:
            st.metric(
                label=MODEL_LABELS[name],
                value=f"{pred['pic50']:.2f} pIC50",
                help=f"90% interval: [{pred['lower']:.2f}, {pred['upper']:.2f}]",
            )
            st.caption(f"90% interval: {pred['lower']:.2f} to {pred['upper']:.2f}")

    if len(models) > 1:
        values = [result["predictions"][name]["pic50"] for name in models]
        spread = max(values) - min(values)
        comparison_text = (
            f"**Model comparison** — predictions span {spread:.2f} pIC50 units"
        )
        if spread < 0.5:
            st.success(f"{comparison_text} (consistent).", icon="✅")
        else:
            st.warning(
                f"{comparison_text} (notable disagreement, worth a closer look).",
                icon="⚠️",
            )
        render_comparison_chart(result, models)


st.sidebar.title("\U0001f9ec EGFR pIC50 Prediction")
st.sidebar.markdown("---")

all_models = list(MODEL_LABELS)
ALL_THREE_LABEL = "Compare all three"
model_dropdown_options = [MODEL_LABELS[m] for m in all_models] + [ALL_THREE_LABEL]
selected_label = st.sidebar.selectbox(
    "Model",
    options=model_dropdown_options,
    index=len(model_dropdown_options) - 1,  # default to "Compare all three"
    help="Run a single model, or compare all three on the same molecule.",
)
selected_models = (
    all_models
    if selected_label == ALL_THREE_LABEL
    else [m for m in all_models if MODEL_LABELS[m] == selected_label]
)

use_api = False
if API_URL:
    # Only offered when API_URL is explicitly set (the docker-compose
    # dual-container setup) -- single-container deploys (Render, Streamlit
    # Cloud, HF Spaces) have no API server to call.
    use_api = st.sidebar.checkbox(
        "Use API backend",
        value=False,
        help="Call a running FastAPI server instead of predicting in-process.",
    )
    if use_api:
        st.sidebar.caption(f"API URL: {API_URL}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Models")
for name in all_models:
    desc = MODEL_DESCRIPTIONS[name]
    st.sidebar.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{MODEL_LABELS[name]}</div>
            <div class="model-tagline">{desc["tagline"]}</div>
            <div class="model-detail">{desc["detail"]}</div>
            <div class="model-rmse">Test RMSE: {desc["rmse"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Trained on ChEMBL bioactivity data with a scaffold split held out for "
    "evaluation. Every prediction carries an uncertainty interval and an "
    "applicability domain flag -- see the project README for methodology "
    "details."
)

st.markdown(
    """
    <div class="app-header">
        <h1>EGFR pIC50 Prediction</h1>
        <p>pIC50 = -log10(IC50 in molar).
           Higher pIC50 means stronger predicted binding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch = st.tabs(["Single molecule", "Batch prediction"])

with tab_single:
    col_input, col_mol = st.columns([2, 1])

    with col_input:
        example_choice = st.selectbox("Example molecule", list(EXAMPLE_MOLECULES))
        default_smiles = EXAMPLE_MOLECULES[example_choice] or (
            "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
        )
        smiles_input = st.text_input("SMILES", value=default_smiles)
        predict_clicked = st.button("Predict pIC50", type="primary", width="stretch")

    with col_mol:
        mol_img = draw_molecule(smiles_input) if smiles_input else None
        if mol_img is not None:
            st.image(mol_img, caption="Molecule structure")
        elif smiles_input:
            st.warning("RDKit could not parse this SMILES.")

    if predict_clicked and smiles_input:
        n_chemprop = sum(m in {"chemeleon", "chemprop"} for m in selected_models)
        spinner_note = (
            f" (Chemprop-family models take ~10-15s each; {n_chemprop} selected)"
            if n_chemprop
            else ""
        )
        with st.spinner(f"Predicting...{spinner_note}"):
            try:
                results = run_predictions([smiles_input], selected_models, use_api)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                results = None

        if results is not None:
            result = results[0]
            if "error" in result:
                st.error(result["error"])
            else:
                render_prediction_cards(result, selected_models)

                history = st.session_state.setdefault("history", [])
                history.insert(
                    0,
                    {
                        "smiles": smiles_input,
                        "models": ", ".join(MODEL_LABELS[m] for m in selected_models),
                        **{
                            f"{MODEL_LABELS[name]} pIC50": round(
                                result["predictions"][name]["pic50"], 2
                            )
                            for name in selected_models
                        },
                        "in_domain": result["applicability_domain"]["in_domain"],
                    },
                )
                st.session_state["history"] = history[:10]

    if st.session_state.get("history"):
        st.markdown("---")
        st.markdown("**Recent predictions** (this session)")
        st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

with tab_batch:
    input_method = st.radio(
        "Input method", ["Paste SMILES", "Upload CSV"], horizontal=True
    )

    smiles_list: list[str] = []
    if input_method == "Paste SMILES":
        text = st.text_area(
            "One SMILES per line",
            value="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1\nCC(=O)Oc1ccccc1C(=O)O",
            height=150,
        )
        smiles_list = [s.strip() for s in text.strip().split("\n") if s.strip()]
    else:
        uploaded = st.file_uploader("CSV with a 'smiles' column", type=["csv"])
        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            smiles_col = st.selectbox(
                "SMILES column",
                df_upload.columns.tolist(),
                index=df_upload.columns.tolist().index("smiles")
                if "smiles" in df_upload.columns
                else 0,
            )
            smiles_list = df_upload[smiles_col].astype(str).tolist()

    if st.button("Predict all", type="primary", disabled=not smiles_list):
        with st.spinner(f"Predicting {len(smiles_list)} molecules..."):
            try:
                results = run_predictions(smiles_list, selected_models, use_api)
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
                results = None

        if results is not None:
            rows = []
            for r in results:
                row = {"smiles": r["smiles"]}
                if "error" in r:
                    row["error"] = r["error"]
                else:
                    row["in_domain"] = r["applicability_domain"]["in_domain"]
                    row["nn_similarity"] = round(
                        r["applicability_domain"]["nearest_neighbor_similarity"], 3
                    )
                    for name in selected_models:
                        row[f"{name}_pic50"] = round(r["predictions"][name]["pic50"], 3)
                rows.append(row)

            results_df = pd.DataFrame(rows)
            st.dataframe(results_df, width="stretch")
            st.download_button(
                "Download results (CSV)",
                data=results_df.to_csv(index=False),
                file_name="egfr_pic50_predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "Random Forest and Chemprop D-MPNN test-set metrics are reported in the "
    "project README. Predictions here are for portfolio demonstration, not "
    "for real drug discovery decisions."
)
