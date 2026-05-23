import pickle
import sqlite3

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

con = sqlite3.connect("smartfeed.db")
df = pd.read_sql("SELECT * FROM interactions", con)
con.close()

# --- feature engineering ---
user_ctr = df.groupby("user_id")["clicked"].mean().rename("user_ctr")
cat_ctr = df.groupby("category")["clicked"].mean().rename("cat_ctr")

df = df.join(user_ctr, on="user_id").join(cat_ctr, on="category")

category_order = ["electronics", "food", "fashion", "sports", "home"]
df["category_enc"] = df["category"].map({c: i for i, c in enumerate(category_order)})

FEATURES = ["user_id", "item_id", "category_enc", "price_rank", "recency_days", "user_ctr", "cat_ctr"]
TARGET = "clicked"

# time-based split: first 80% train, last 20% test (preserves row order as proxy for time)
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {auc:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("Saved model.pkl and features.pkl")
