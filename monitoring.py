import pickle
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

DB = "smartfeed.db"
N_BINS = 10
PSI_WARN = 0.10
PSI_ALERT = 0.20
CATEGORY_ORDER = ["electronics", "food", "fashion", "sports", "home"]
MONITOR_INTERVAL_HOURS = 6   # how often PSI is checked
RETRAIN_COOLDOWN_HOURS = 12  # minimum gap between two retrains


# ---------- data loading ----------

def _load_reference_scores() -> np.ndarray:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM interactions", con)
    con.close()

    user_ctr = df.groupby("user_id")["clicked"].mean().rename("user_ctr")
    cat_ctr  = df.groupby("category")["clicked"].mean().rename("cat_ctr")
    df = df.join(user_ctr, on="user_id").join(cat_ctr, on="category")
    df["category_enc"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})

    return model.predict_proba(df[features])[:, 1]


def _load_live_scores() -> np.ndarray:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT score FROM predictions").fetchall()
    con.close()
    return np.array([r["score"] for r in rows], dtype=float)


# ---------- PSI ----------

def compute_psi(reference: np.ndarray, actual: np.ndarray) -> tuple[float, list[dict]]:
    bins = np.linspace(0, 1, N_BINS + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    ref_pct = ref_counts / ref_counts.sum()
    act_pct = act_counts / act_counts.sum()

    ref_pct = np.clip(ref_pct, 1e-8, None)
    act_pct = np.clip(act_pct, 1e-8, None)

    psi_per_bin = (act_pct - ref_pct) * np.log(act_pct / ref_pct)

    buckets = [
        {
            "label":        f"[{bins[i]:.1f} – {bins[i+1]:.1f}]",
            "expected_pct": ref_pct[i],
            "actual_pct":   act_pct[i],
            "contribution": psi_per_bin[i],
        }
        for i in range(N_BINS)
    ]

    return float(psi_per_bin.sum()), buckets


# ---------- report ----------

def run() -> float | None:
    """Print PSI report. Returns PSI value, or None if there were not enough live samples."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 64)
    print("  SmartFeed Model Monitoring Report")
    print(f"  Generated : {now}")
    print("=" * 64)

    reference = _load_reference_scores()
    live      = _load_live_scores()

    print(f"\n  Reference : {len(reference):>8,} samples  (training data scored with current model)")
    print(f"  Live      : {len(live):>8,} samples  (predictions table)")

    if len(live) == 0:
        print("\n  WARNING: predictions table is empty — nothing to compare against.")
        print("  Call POST /rank a few times then re-run.")
        print("=" * 64)
        return None

    if len(live) < 30:
        print(f"\n  NOTE: Only {len(live)} live samples — PSI may be noisy. Recommend 30+.")

    psi, buckets = compute_psi(reference, live)

    print(f"\n  {'Bucket':<14}  {'Expected %':>10}  {'Actual %':>10}  {'PSI Contrib':>12}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*12}")
    for b in buckets:
        print(
            f"  {b['label']:<14}  "
            f"{b['expected_pct'] * 100:>9.2f}%  "
            f"{b['actual_pct'] * 100:>9.2f}%  "
            f"{b['contribution']:>12.5f}"
        )

    print(f"\n  {'PSI Score':<14}  {psi:.5f}")
    print()

    if psi < PSI_WARN:
        verdict = "STABLE"
        note    = f"PSI < {PSI_WARN:.2f}  —  no significant distribution shift"
    elif psi < PSI_ALERT:
        verdict = "MONITOR"
        note    = f"PSI {PSI_WARN:.2f}–{PSI_ALERT:.2f}  —  minor shift, keep watching"
    else:
        verdict = "*** DRIFT DETECTED ***"
        note    = f"PSI > {PSI_ALERT:.2f}  —  triggering retrain"

    print(f"  Status    : {verdict}")
    print(f"  Detail    : {note}")
    print("=" * 64)

    return psi


# ---------- cooldown guard ----------

def _on_cooldown() -> bool:
    """Return True if a retrain ran less than RETRAIN_COOLDOWN_HOURS ago."""
    import json
    from pathlib import Path

    path = Path("model_metadata.json")
    if not path.exists():
        return False

    with open(path) as f:
        meta = json.load(f)

    trained_at = meta.get("trained_at")
    if not trained_at:
        return False

    last = datetime.fromisoformat(trained_at)
    elapsed_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return elapsed_hours < RETRAIN_COOLDOWN_HOURS


# ---------- the joined job ----------

def check_and_retrain() -> None:
    """Run PSI check. If drift is detected and cooldown has passed, trigger retrain."""
    from retrain import retrain

    psi = run()

    if psi is None:
        return

    if psi > PSI_ALERT:
        if _on_cooldown():
            print(f"\n  Drift detected but retrain ran < {RETRAIN_COOLDOWN_HOURS}h ago — skipping.")
            print(f"  Next window opens in ~{RETRAIN_COOLDOWN_HOURS}h.\n")
        else:
            print("\n  Drift threshold exceeded — triggering retrain now...\n")
            retrain()
    else:
        print("\n  No retrain needed.\n")


# ---------- entry point ----------

if __name__ == "__main__":
    # Run one check immediately so you see output right away
    check_and_retrain()

    scheduler = BlockingScheduler(timezone="UTC")

    # Primary: PSI-driven retrain every MONITOR_INTERVAL_HOURS
    scheduler.add_job(
        check_and_retrain,
        "interval",
        hours=MONITOR_INTERVAL_HOURS,
        id="psi_check",
    )

    # Safety net: retrain every Sunday at 02:00 UTC regardless of PSI,
    # so the model stays fresh even if drift never crosses the alert threshold
    scheduler.add_job(
        __import__("retrain").retrain,
        "cron",
        day_of_week="sun",
        hour=2,
        id="weekly_retrain",
    )

    print(f"\nScheduler running.")
    print(f"  PSI check    : every {MONITOR_INTERVAL_HOURS}h  (retrain if PSI > {PSI_ALERT})")
    print(f"  Retrain cooldown : {RETRAIN_COOLDOWN_HOURS}h between consecutive retrains")
    print(f"  Safety retrain   : every Sunday 02:00 UTC")
    print("Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
