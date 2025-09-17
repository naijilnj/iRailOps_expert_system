# train_delay_model.py
"""
Robust training script for delay classifier (fixed to encode string labels).
- Drops 'Unknown' label rows by default (set DROP_UNKNOWN=False to keep).
- Saves:
    - delay_xgb.joblib  (model)
    - encoders.joblib   (dict with LabelEncoders for categorical fields + label encoder)

Usage:
  python train_delay_model.py
"""

import os
import joblib
import pandas as pd
import traceback
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import psycopg2

# CONFIG (override with env vars if needed)
DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

MODEL_OUT = os.getenv("MODEL_FILE", "delay_xgb.joblib")
ENC_OUT = os.getenv("ENCODERS_FILE", "encoders.joblib")

TABLE_NAME = os.getenv("DELAY_TABLE", "train_delays")
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "500000"))

# When True drop rows with cause_label == 'Unknown' (recommended)
DROP_UNKNOWN = True
MIN_ROWS = 50  # warn if too few rows after filtering

def get_table_columns(conn, table_name):
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = %s
    """
    cur = conn.cursor()
    cur.execute(q, (table_name,))
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    return cols

def fetch_delay_data(conn, table_name, cols):
    # build select list from available columns
    select_cols = ", ".join(cols)
    q = f"SELECT {select_cols} FROM {table_name} LIMIT {ROW_LIMIT}"
    df = pd.read_sql(q, conn)
    return df

def build_features(df):
    df = df.copy()

    # Ensure expected columns exist, otherwise create safe defaults
    if "seq" not in df.columns:
        df["seq"] = -1
    if "distance" not in df.columns:
        df["distance"] = 0
    if "delay_minutes" not in df.columns:
        for alt in ["delay_min", "delay", "delay_mins"]:
            if alt in df.columns:
                df["delay_minutes"] = pd.to_numeric(df[alt], errors="coerce").fillna(0).astype(int)
                break
        else:
            df["delay_minutes"] = 0

    # categorical features: train_no, station_code, source_station, destination_station
    cat_cols = []
    for c in ["train_no", "station_code", "source_station", "destination_station"]:
        if c not in df.columns:
            df[c] = "NA"
        else:
            df[c] = df[c].astype(str).fillna("NA")
        cat_cols.append(c)

    if "cause_label" not in df.columns:
        raise ValueError("target column 'cause_label' not found in train_delays table — training cannot proceed")

    # Optionally drop Unknowns
    if DROP_UNKNOWN and "cause_label" in df.columns:
        before = len(df)
        df = df[df["cause_label"].astype(str).str.strip().str.lower() != "unknown"]
        after = len(df)
        print(f"Dropped {before-after} rows with cause_label='Unknown'. Rows remaining: {after}")

    if len(df) < MIN_ROWS:
        print(f"Warning: only {len(df)} rows available after filtering. Model quality will be poor.")

    # encode categorical via LabelEncoder (save encoders)
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = df[c].fillna("NA").astype(str)
        le.fit(df[c].values)
        df[c + "_enc"] = le.transform(df[c].values)
        encoders[c] = le

    # build X and y
    feature_cols = ["train_no_enc", "station_code_enc", "seq", "distance", "delay_minutes"]
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(-1).astype(int)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int)
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0).astype(int)

    X = df[feature_cols].copy()
    y = df["cause_label"].astype(str).values

    return X, y, encoders

def train_model(X, y):
    # Label-encode y (string classes -> integers)
    y_le = LabelEncoder()
    y_enc = y_le.fit_transform(y)
    print("Classes:", list(y_le.classes_))

    # Try xgboost first
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=42)
        print("Using XGBoost for training")
    except Exception:
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        print("XGBoost not available; using RandomForest")

    test_size = 0.2 if len(y_enc) >= 50 else 0.3
    stratify = y_enc if len(y_enc) >= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42, stratify=stratify)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classification report on test set (integer labels):")
    print(classification_report(y_test, preds, target_names=list(y_le.classes_)))
    return model, y_le

def main():
    conn = None
    try:
        print("Connecting to Postgres...")
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        cols = get_table_columns(conn, TABLE_NAME)
        if not cols:
            print(f"Table {TABLE_NAME} not found or has no columns. Check DB and schema.")
            return
        print(f"Found columns in {TABLE_NAME}: {cols}")

        if "cause_label" not in cols:
            print("ERROR: 'cause_label' (target) not present in train_delays table. Add this column or populate it.")
            return

        print("Fetching rows (limited)...")
        df = fetch_delay_data(conn, TABLE_NAME, cols)
        print(f"Rows fetched: {df.shape[0]}")

        X, y, encoders = build_features(df)
        print("Training model...")
        model, label_encoder = train_model(X, y)

        print(f"Saving model to {MODEL_OUT}")
        joblib.dump(model, MODEL_OUT)

        # Save encoders + label encoder
        all_encoders = {
            "categorical_encoders": encoders,
            "label_encoder": label_encoder
        }
        print(f"Saving encoders to {ENC_OUT}")
        joblib.dump(all_encoders, ENC_OUT)

        print("Done.")
    except Exception as e:
        traceback.print_exc()
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    main()
