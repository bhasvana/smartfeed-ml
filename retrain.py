import json
import logging
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

DB = "smartfeed.db"
MODEL_FILE = "model.pkl"
FEATURES_FILE = "features.pkl"
METADATA_FILE = "model_metadata.json"
MIN_AUC_IMPROVEMENT = 0.02
CATEGORY_ORDER = ["electronics", "food", "fashion", "sports", "home"]
FEATURES = ["user_id", "item_id", "category_enc", "price_rank", "recency_days", "user_ctr", "cat_ctr"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------- metadata ----------

def _read_metadata() -> dict:
    path = Path(METADATA_FILE)
    if not path.exists():
        # No metadata yet — treat current model as having 0.0 AUC so first retrain always seeds the file
        return {"auc": 0.0, "trained_at": None, "n_samples": 0, "model_version": 0}
    with open(path) as f:
        return json.load(f)


def _write_metadata(auc: float, n_samples: int, version: int) -> None:
    payload = {
        "auc":           round(auc, 6),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_samples":     n_samples,
        "model_version": version,
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s  (version=%d  auc=%.4f)", METADATA_FILE, version, auc)


# ---------- feature engineering ----------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must stay identical to train.py."""
    user_ctr = df.groupby("user_id")["clicked"].mean().rename("user_ctr")
    cat_ctr  = df.groupby("category")["clicked"].mean().rename("cat_ctr")
    df = df.join(user_ctr, on="user_id").join(cat_ctr, on="category")
    df["category_enc"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    return df


# ---------- retrain ----------

def retrain() -> None:
    print("\n" + "=" * 64)
    print(f"  Retrain started  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 64)

    # --- load interactions (grows over time as new clicks are logged) ---
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM interactions", con)
    con.close()

    n = len(df)
    print(f"\n  Loaded {n:,} rows from interactions table")

    if n < 1000:
        print("  ABORTED: fewer than 1,000 interactions — not enough data to retrain reliably.")
        print("=" * 64)
        return

    df = _engineer_features(df)

    # time-based split — no shuffle, same as train.py
    split = int(n * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train, y_train = train_df[FEATURES], train_df["clicked"]
    X_test,  y_test  = test_df[FEATURES],  test_df["clicked"]

    # --- train ---
    print(f"  Training on {len(train_df):,} rows, validating on {len(test_df):,} rows...")
    new_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        verbosity=-1,
    )
    new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    new_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])
    print(f"\n  New model AUC     : {new_auc:.4f}")

    # --- compare against current model ---
    meta    = _read_metadata()
    old_auc = meta["auc"]
    version = meta["model_version"]

    trained_at = meta["trained_at"] or "never"
    print(f"  Current model AUC : {old_auc:.4f}  (v{version}, trained {trained_at})")

    improvement = new_auc - old_auc
    print(f"  Improvement       : {improvement:+.4f}  (required > +{MIN_AUC_IMPROVEMENT})")

    # --- decide ---
    if improvement > MIN_AUC_IMPROVEMENT:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(new_model, f)
        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(FEATURES, f)

        _write_metadata(new_auc, n, version + 1)

        print(f"\n  SUCCESS: model.pkl replaced.")
        print(f"           v{version} → v{version + 1}   AUC {old_auc:.4f} → {new_auc:.4f}")
        print("           Restart api.py to load the new model into memory.")
    else:
        print(f"\n  NO CHANGE: improvement ({improvement:+.4f}) did not exceed threshold (+{MIN_AUC_IMPROVEMENT}).")
        print("             Keeping current model.")

    print("=" * 64 + "\n")


if __name__ == "__main__":
    # Run a one-shot retrain manually.
    # Scheduling is owned by monitoring.py.
    retrain()
