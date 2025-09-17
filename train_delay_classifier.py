# train_delay_classifier.py
"""
Train a RandomForest delay-cause classifier from `train_delays` + `train_timetable`,
save the trained model and categorical mappings using joblib.
"""

import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sqlalchemy import create_engine

# ---------- CONFIG ----------
DB_URL = "postgresql://postgres:142003@localhost:5432/railway"  # change if needed
MODEL_FILE = "delay_classifier.joblib"
MAPPINGS_FILE = "cat_mappings.joblib"
# ----------------------------

print("📥 Loading data from DB...")
engine = create_engine(DB_URL)

query = """
SELECT d.train_no, d.station_code, d.delay_minutes, d.delay_reason,
       t.seq, COALESCE(t.distance, 0) as distance,
       t.source_station, t.destination_station
FROM train_delays d
JOIN train_timetable t
  ON d.train_no = t.train_no AND d.station_code = t.station_code;
"""
df = pd.read_sql(query, engine)
print("✅ Rows loaded:", len(df))

# Basic cleaning
df = df.dropna(subset=["delay_minutes", "delay_reason"])
df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0).astype(int)
df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(-1).astype(int)
df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int)

# Ensure delay_reason is categorical
df["delay_reason"] = df["delay_reason"].astype(str)
df["delay_reason"] = df["delay_reason"].str.strip()

# Create categorical mappings and encode
cat_columns = ["train_no", "station_code", "source_station", "destination_station"]
mappings = {}

for col in cat_columns:
    df[col] = df[col].fillna("UNKNOWN").astype(str)
    cats = pd.Categorical(df[col])
    categories = list(cats.categories)
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    df[col + "_code"] = df[col].map(mapping).fillna(-1).astype(int)
    mappings[col] = mapping

# Prepare features and target
feature_cols = [
    "train_no_code", "station_code_code", "seq", "distance", "delay_minutes",
    "source_station_code", "destination_station_code"
]

# rename the encoded columns for convenience
df = df.rename(columns={
    "train_no_code": "train_no_code",
    "station_code_code": "station_code_code",
    "source_station_code": "source_station_code",
    "destination_station_code": "destination_station_code"
})

# if the rename didn't create those names (safe fallback)
if "train_no_code" not in df.columns:
    df["train_no_code"] = df["train_no"].map(mappings["train_no"]).fillna(-1).astype(int)
if "station_code_code" not in df.columns:
    df["station_code_code"] = df["station_code"].map(mappings["station_code"]).fillna(-1).astype(int)
if "source_station_code" not in df.columns:
    df["source_station_code"] = df["source_station"].map(mappings["source_station"]).fillna(-1).astype(int)
if "destination_station_code" not in df.columns:
    df["destination_station_code"] = df["destination_station"].map(mappings["destination_station"]).fillna(-1).astype(int)

X = df[["train_no_code", "station_code_code", "seq", "distance", "delay_minutes",
        "source_station_code", "destination_station_code"]]
y = df["delay_reason"]

# Train/test split
print("🚀 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and mappings
joblib.dump(clf, MODEL_FILE)
joblib.dump(mappings, MAPPINGS_FILE)
print(f"✅ Model saved to {MODEL_FILE}")
print(f"✅ Categorical mappings saved to {MAPPINGS_FILE}")
