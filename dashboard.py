import json
import pickle
import sqlite3

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DB = "smartfeed.db"
MODEL_FILE = "model.pkl"
FEATURES_FILE = "features.pkl"
METADATA_FILE = "model_metadata.json"
CATEGORY_ORDER = ["electronics", "food", "fashion", "sports", "home"]
N_BINS = 10
PSI_ALERT = 0.20

st.set_page_config(page_title="SmartFeed Dashboard", layout="wide")
st.title("SmartFeed — ML Dashboard")


# ============================================================
# Loaders  (cached so reruns don't re-read disk/DB)
# ============================================================

@st.cache_resource
def _model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def _features():
    with open(FEATURES_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=60)
def _metadata() -> dict:
    with open(METADATA_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def _recent_predictions() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        """
        SELECT created_at AS timestamp,
               user_id,
               item_id    AS top_item_id,
               score
        FROM   predictions
        WHERE  rank = 1
        ORDER  BY id DESC
        LIMIT  100
        """,
        con,
    )
    con.close()
    return df


@st.cache_data(ttl=300)
def _reference_scores() -> np.ndarray:
    model    = _model()
    features = _features()
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM interactions", con)
    con.close()

    user_ctr = df.groupby("user_id")["clicked"].mean().rename("user_ctr")
    cat_ctr  = df.groupby("category")["clicked"].mean().rename("cat_ctr")
    df = df.join(user_ctr, on="user_id").join(cat_ctr, on="category")
    df["category_enc"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})

    return model.predict_proba(df[features])[:, 1]


@st.cache_data(ttl=60)
def _daily_psi() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT score, created_at FROM predictions", con)
    con.close()

    if df.empty:
        return pd.DataFrame(columns=["date", "psi"])

    df["date"] = pd.to_datetime(df["created_at"]).dt.date

    reference = _reference_scores()
    bins = np.linspace(0, 1, N_BINS + 1)
    ref_counts, _ = np.histogram(reference, bins=bins)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-8, None)

    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        act_counts, _ = np.histogram(grp["score"].values, bins=bins)
        act_pct = np.clip(act_counts / act_counts.sum(), 1e-8, None)
        psi = float(((act_pct - ref_pct) * np.log(act_pct / ref_pct)).sum())
        rows.append({"date": str(date), "psi": round(psi, 5)})

    return pd.DataFrame(rows)


# ============================================================
# Section 1 — Model Health
# ============================================================

st.header("1 — Model Health")

meta = _metadata()

c1, c2, c3 = st.columns(3)
c1.metric("AUC", f"{meta['auc']:.4f}")
c2.metric("Model Version", f"v{meta['model_version']}")
c3.metric("Training Samples", f"{meta['n_samples']:,}")
st.caption(f"Last trained: {meta.get('trained_at', 'N/A')}")

features = _features()
importances = _model().feature_importances_

imp_df = (
    pd.DataFrame({"feature": features, "importance": importances})
    .sort_values("importance")
)

imp_chart = (
    alt.Chart(imp_df)
    .mark_bar()
    .encode(
        x=alt.X("importance:Q", title="Importance (split count)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=["feature", "importance"],
    )
    .properties(height=220)
)
st.altair_chart(imp_chart, use_container_width=True)

st.divider()


# ============================================================
# Section 2 — Recent Predictions
# ============================================================

st.header("2 — Recent Predictions")
st.caption("Top-ranked item (rank = 1) per request, last 100 requests.")

pred_df = _recent_predictions()
if pred_df.empty:
    st.info("No predictions yet — call POST /rank a few times first.")
else:
    st.dataframe(pred_df, use_container_width=True)

st.divider()


# ============================================================
# Section 3 — Drift Monitor
# ============================================================

st.header("3 — Drift Monitor")

psi_df = _daily_psi()

if psi_df.empty:
    st.info("Not enough prediction data to compute PSI. Need ≥ 10 predictions per day.")
else:
    latest_psi = float(psi_df.iloc[-1]["psi"])

    if latest_psi > PSI_ALERT:
        st.error(
            f"DRIFT DETECTED — latest daily PSI = {latest_psi:.4f}  "
            f"(alert threshold: {PSI_ALERT})"
        )
    else:
        st.success(
            f"Stable — latest daily PSI = {latest_psi:.4f}  "
            f"(alert threshold: {PSI_ALERT})"
        )

    psi_df["date"] = pd.to_datetime(psi_df["date"])

    line = (
        alt.Chart(psi_df)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("psi:Q", title="Daily Average PSI", scale=alt.Scale(domainMin=0)),
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "psi:Q"],
        )
    )
    threshold_line = (
        alt.Chart(pd.DataFrame({"y": [PSI_ALERT]}))
        .mark_rule(color="red", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    threshold_label = (
        alt.Chart(pd.DataFrame({"y": [PSI_ALERT], "label": [f"Alert ({PSI_ALERT})"]}))
        .mark_text(align="left", dx=4, dy=-6, color="red", fontSize=11)
        .encode(
            y=alt.Y("y:Q"),
            x=alt.value(0),
            text="label:N",
        )
    )

    st.altair_chart(line + threshold_line + threshold_label, use_container_width=True)

st.divider()


# ============================================================
# Section 4 — Retraining History
# ============================================================

st.header("4 — Retraining History")
st.caption("Sourced from model_metadata.json — one row per promoted model version.")

history_df = pd.DataFrame(
    [
        {
            "version":    f"v{meta['model_version']}",
            "auc":        meta["auc"],
            "n_samples":  meta["n_samples"],
            "trained_at": meta.get("trained_at", "N/A"),
        }
    ]
)
st.dataframe(history_df, use_container_width=True)
