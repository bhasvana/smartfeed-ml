import sqlite3
import numpy as np
import pandas as pd

np.random.seed(42)

N = 100_000
categories = ["electronics", "food", "fashion", "sports", "home"]

user_ids = np.random.randint(1, 1001, N)
item_ids = np.random.randint(1, 501, N)
category = np.random.choice(categories, N)
price_rank = np.random.randint(1, 11, N)
recency_days = np.random.randint(0, 31, N)

# Signal centred at 0 so mean CTR tracks the base logit
signal = (
    0.02 * (10 - price_rank)        # mean contribution ~0.09
    + 0.02 * (30 - recency_days)    # mean contribution ~0.30
    + np.where(category == "electronics", 0.15, 0.0)
    + np.where(category == "food", -0.10, 0.0)
)
signal -= signal.mean()  # centre so base_logit controls the overall CTR

# ln(0.15 / 0.85) ≈ -1.735 → ~15 % CTR
base_logit = np.log(0.15 / 0.85)
prob = 1 / (1 + np.exp(-(base_logit + signal)))
clicked = (np.random.random(N) < prob).astype(int)

df = pd.DataFrame(
    {
        "user_id": user_ids,
        "item_id": item_ids,
        "category": category,
        "price_rank": price_rank,
        "recency_days": recency_days,
        "clicked": clicked,
    }
)

print(f"Overall CTR: {df['clicked'].mean():.3f}")

con = sqlite3.connect("smartfeed.db")
df.to_sql("interactions", con, if_exists="replace", index=False)
con.close()

print(f"Saved {N} rows to smartfeed.db")
