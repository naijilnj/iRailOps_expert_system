import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# ---------------- CONFIG ----------------
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASS = "142003"
DB_HOST = "localhost"
DB_PORT = "5432"
# ----------------------------------------

# Connect to DB
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
)
cur = conn.cursor()

# Fetch timetable
df = pd.read_sql("SELECT train_no, station_code, arrival_time FROM train_timetable ORDER BY train_no, seq", conn)

# Clean arrival_time to datetime
df["scheduled_arrival"] = pd.to_datetime(df["arrival_time"], errors="coerce", format="%H:%M:%S")

# Initialize delay
np.random.seed(42)
df["delay_minutes"] = np.random.randint(0, 60, size=len(df))

# Propagate delay downstream (cumulative within each train)
df["delay_minutes"] = df.groupby("train_no")["delay_minutes"].cumsum()

# Compute actual arrival
df["actual_arrival"] = df["scheduled_arrival"] + pd.to_timedelta(df["delay_minutes"], unit="m")

# Assign mock reasons
reasons = ["Infrastructure", "Congestion", "Operational", "External"]
probs = [0.4, 0.3, 0.2, 0.1]
df["delay_reason"] = np.random.choice(reasons, size=len(df), p=probs)

# ---------------- CREATE TABLE ----------------
create_table_sql = """
CREATE TABLE IF NOT EXISTS train_delays (
    id SERIAL PRIMARY KEY,
    train_no VARCHAR(20),
    station_code VARCHAR(20),
    scheduled_arrival TIME,
    actual_arrival TIME,
    delay_minutes INT,
    delay_reason VARCHAR(50)
);
"""
cur.execute(create_table_sql)
conn.commit()

# ---------------- INSERT DATA ----------------
insert_sql = """
INSERT INTO train_delays (
    train_no, station_code, scheduled_arrival, actual_arrival, delay_minutes, delay_reason
) VALUES %s
"""

values = [
    (
        row.train_no,
        row.station_code,
        row.scheduled_arrival.time() if pd.notna(row.scheduled_arrival) else None,
        row.actual_arrival.time() if pd.notna(row.actual_arrival) else None,
        int(row.delay_minutes) if pd.notna(row.delay_minutes) else None,
        row.delay_reason
    )
    for row in df.itertuples(index=False)
]

execute_values(cur, insert_sql, values)
conn.commit()

print(f"✅ Inserted {len(values)} simulated delay rows into train_delays")

cur.close()
conn.close()
