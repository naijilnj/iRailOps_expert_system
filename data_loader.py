import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---------------- CONFIG ----------------
CSV_FILE = "train_rout..csv"   # your file
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASS = "142003"   # change if needed
DB_HOST = "localhost"
DB_PORT = "5432"
# ----------------------------------------

print("Loading CSV...")
df = pd.read_csv(CSV_FILE, dtype=str)

# Clean column names: lowercase, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Replace Source/Destination text in arrival_time
if "arrival_time" in df.columns:
    df["arrival_time"] = df["arrival_time"].replace({"Source": None, "Destination": None})

print("✅ Columns after cleaning:", df.columns.tolist())

# Convert NaN to None for DB
df = df.where(pd.notnull(df), None)

# Explicit column order to match SQL insert
columns_order = [
    "train_no", "train_name", "seq", "station_code", "station_name",
    "arrival_time", "departure_time", "distance",
    "source_station", "source_station_name",
    "destination_station", "destination_station_name"
]

df = df[columns_order]

# Convert numeric fields safely
df["seq"] = pd.to_numeric(df["seq"], errors="coerce").astype("Int64")
df["distance"] = pd.to_numeric(df["distance"], errors="coerce").astype("Int64")

# ---------------- CREATE TABLE ----------------
create_table_sql = """
CREATE TABLE IF NOT EXISTS train_timetable (
    id SERIAL PRIMARY KEY,
    train_no VARCHAR(20),
    train_name VARCHAR(255),
    seq INT,
    station_code VARCHAR(20),
    station_name VARCHAR(255),
    arrival_time VARCHAR(20),
    departure_time VARCHAR(20),
    distance INT,
    source_station VARCHAR(20),
    source_station_name VARCHAR(255),
    destination_station VARCHAR(20),
    destination_station_name VARCHAR(255)
);
"""

# ---------------- DB INSERT ----------------
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
)
cur = conn.cursor()

# Create table
cur.execute(create_table_sql)
conn.commit()

# Insert data
insert_sql = """
INSERT INTO train_timetable (
    train_no, train_name, seq, station_code, station_name,
    arrival_time, departure_time, distance,
    source_station, source_station_name,
    destination_station, destination_station_name
) VALUES %s
"""

# Convert DataFrame to list of tuples with native Python types
values = [
    (
        str(row.train_no) if row.train_no is not None else None,
        str(row.train_name) if row.train_name is not None else None,
        int(row.seq) if pd.notna(row.seq) else None,
        str(row.station_code) if row.station_code is not None else None,
        str(row.station_name) if row.station_name is not None else None,
        str(row.arrival_time) if row.arrival_time is not None else None,
        str(row.departure_time) if row.departure_time is not None else None,
        int(row.distance) if pd.notna(row.distance) else None,
        str(row.source_station) if row.source_station is not None else None,
        str(row.source_station_name) if row.source_station_name is not None else None,
        str(row.destination_station) if row.destination_station is not None else None,
        str(row.destination_station_name) if row.destination_station_name is not None else None,
    )
    for row in df.itertuples(index=False)
]

# Debug: preview first 5 rows before insert
print("🔎 Preview first 5 rows:", values[:5])

execute_values(cur, insert_sql, values)
conn.commit()

print(f"✅ Inserted {len(values)} rows into train_timetable")

cur.close()
conn.close()
