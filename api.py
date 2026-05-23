import pickle
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB = "smartfeed.db"
CATEGORY_ORDER = ["electronics", "food", "fashion", "sports", "home"]
CAT_ENC = {c: i for i, c in enumerate(CATEGORY_ORDER)}
FALLBACK_CTR = 0.15

_state: dict = {}


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def _init_predictions_table() -> None:
    with _con() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                item_id     INTEGER NOT NULL,
                score       REAL    NOT NULL,
                rank        INTEGER NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )


def _load_cat_ctrs() -> dict[str, float]:
    with _con() as con:
        rows = con.execute(
            "SELECT category, AVG(clicked) AS ctr FROM interactions GROUP BY category"
        ).fetchall()
    return {r["category"]: r["ctr"] for r in rows}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("model.pkl", "rb") as f:
        _state["model"] = pickle.load(f)
    with open("features.pkl", "rb") as f:
        _state["features"] = pickle.load(f)
    _state["cat_ctrs"] = _load_cat_ctrs()
    _init_predictions_table()
    yield
    _state.clear()


app = FastAPI(title="SmartFeed Ranking API", lifespan=lifespan)


# ---------- schema ----------

class RankRequest(BaseModel):
    user_id: int
    item_ids: list[int]


class RankedItem(BaseModel):
    item_id: int
    score: float
    rank: int


class RankResponse(BaseModel):
    user_id: int
    ranked_items: list[RankedItem]
    response_time_ms: float


# ---------- helpers ----------

def _fetch_user_ctr(user_id: int) -> float:
    with _con() as con:
        row = con.execute(
            "SELECT AVG(clicked) AS ctr FROM interactions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return row["ctr"] if row["ctr"] is not None else FALLBACK_CTR


def _fetch_item_features(item_ids: list[int]) -> pd.DataFrame:
    """Return one representative row per item_id (last-seen interaction)."""
    placeholders = ",".join("?" * len(item_ids))
    query = f"""
        SELECT i.item_id, i.category, i.price_rank, i.recency_days
        FROM   interactions i
        INNER JOIN (
            SELECT item_id, MAX(rowid) AS max_rowid
            FROM   interactions
            WHERE  item_id IN ({placeholders})
            GROUP  BY item_id
        ) latest ON i.rowid = latest.max_rowid
    """
    with _con() as con:
        return pd.read_sql(query, con, params=item_ids)


def _build_feature_matrix(
    user_id: int,
    item_ids: list[int],
    user_ctr: float,
    item_df: pd.DataFrame,
) -> pd.DataFrame:
    cat_ctrs = _state["cat_ctrs"]
    item_lookup = item_df.set_index("item_id").to_dict("index")

    rows = []
    for iid in item_ids:
        meta = item_lookup.get(iid)
        if meta:
            cat = meta["category"]
            price_rank = meta["price_rank"]
            recency_days = meta["recency_days"]
        else:
            # cold-start defaults (median-ish values)
            cat = "home"
            price_rank = 5
            recency_days = 15

        rows.append(
            {
                "user_id": user_id,
                "item_id": iid,
                "category_enc": CAT_ENC.get(cat, len(CATEGORY_ORDER) - 1),
                "price_rank": price_rank,
                "recency_days": recency_days,
                "user_ctr": user_ctr,
                "cat_ctr": cat_ctrs.get(cat, FALLBACK_CTR),
            }
        )

    return pd.DataFrame(rows)[_state["features"]]


def _log_predictions(user_id: int, ranked: list[RankedItem]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = [(user_id, r.item_id, r.score, r.rank, now) for r in ranked]
    with _con() as con:
        con.executemany(
            "INSERT INTO predictions (user_id, item_id, score, rank, created_at) VALUES (?,?,?,?,?)",
            rows,
        )


# ---------- endpoints ----------

@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    if not req.item_ids:
        raise HTTPException(status_code=422, detail="item_ids must not be empty")

    t0 = time.perf_counter()

    user_ctr = _fetch_user_ctr(req.user_id)
    item_df = _fetch_item_features(req.item_ids)
    X = _build_feature_matrix(req.user_id, req.item_ids, user_ctr, item_df)

    scores = _state["model"].predict_proba(X)[:, 1]
    order = np.argsort(-scores)

    ranked = [
        RankedItem(item_id=req.item_ids[i], score=round(float(scores[i]), 6), rank=int(pos + 1))
        for pos, i in enumerate(order)
    ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    _log_predictions(req.user_id, ranked)

    return RankResponse(
        user_id=req.user_id,
        ranked_items=ranked,
        response_time_ms=elapsed_ms,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
