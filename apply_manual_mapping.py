# apply_manual_mapping.py
"""
Apply manual mapping CSV (produced by suggest_mappings.py) to train_delays.cause_label.
CSV columns: delay_reason,count,suggested_label,final_label (edit)
Run: python apply_manual_mapping.py
"""

import os
import csv
import psycopg2

DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

CSV_FILE = "delay_reason_suggestions.csv"
TABLE = "train_delays"

def main():
    if not os.path.exists(CSV_FILE):
        print("CSV file not found:", CSV_FILE); return
    rows = []
    with open(CSV_FILE, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dr = r.get("delay_reason")
            final = r.get("final_label (edit)") or r.get("final_label") or r.get("suggested_label")
            if final and dr:
                rows.append((final.strip(), dr.strip()))
    if not rows:
        print("No mappings found in CSV. Make sure you filled the 'final_label (edit)' column.")
        return

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    # ensure column exists
    cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS cause_label VARCHAR(128);")
    conn.commit()

    # apply updates one by one (safe) — use prepared statement
    for lab, dr in rows:
        cur.execute(f"""
            UPDATE {TABLE}
            SET cause_label = %s
            WHERE delay_reason = %s
        """, (lab, dr))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Applied {len(rows)} mappings to {TABLE} (exact-match on delay_reason).")

if __name__ == "__main__":
    main()
