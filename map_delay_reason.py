# map_delay_reason.py
"""
Map free-text delay_reason -> cause_label (Operational | Congestion | Infrastructure | External | Unknown)
Run: python map_delay_reason.py
"""

import os
import psycopg2
import re
from collections import Counter

DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
TABLE = os.getenv("DELAY_TABLE", "train_delays")

# mapping rules (ordered: first match wins)
MAPPINGS = [
    ("Operational", [
        r"\bcrew\b", r"\bdriver\b", r"\bloco\b", r"\bengine\b", r"\btechnical\b",
        r"\bbreakdown\b", r"\bsignal fail\b", r"\bsignal\b", r"\bshunting\b",
        r"\btrailer\b", r"\btraction\b"
    ]),
    ("Congestion", [
        r"\blate\b", r"\brunning late\b", r"\bincoming\b", r"\bcongest", r"\btraffic\b",
        r"\bwait for\b", r"\bhold for\b"
    ]),
    ("Infrastructure", [
        r"\btrack\b", r"\bpoint\b", r"\bbridge\b", r"\bpermanent way\b", r"\bohle\b",
        r"\bcatenary\b", r"\boverhead\b", r"\bsignal circuit\b", r"\bpoints\b"
    ]),
    ("External", [
        r"\bfog\b", r"\brain\b", r"\bstorm\b", r"\btrespass\b", r"\banimal\b",
        r"\baccident\b", r"\bpolice\b", r"\broad\b", r"\bweather\b"
    ]),
]

def map_reason(text):
    if text is None:
        return "Unknown"
    t = text.lower()
    for label, patterns in MAPPINGS:
        for pat in patterns:
            if re.search(pat, t):
                return label
    # fallback heuristics
    if re.search(r"\bdelay\b", t) and len(t) < 30:
        return "Operational"
    return "Unknown"

def main():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    # ensure column exists
    cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS cause_label VARCHAR(128);")
    conn.commit()

    # fetch non-null delay_reason rows where cause_label is null or 'Unknown'
    cur.execute(f"""
        SELECT id, delay_reason
        FROM {TABLE}
        WHERE delay_reason IS NOT NULL
    """)
    rows = cur.fetchall()
    print(f"Rows to process: {len(rows)}")

    updates = []
    counts = Counter()
    sample_unknown = []
    for r in rows:
        rid, dr = r
        mapped = map_reason(dr)
        counts[mapped] += 1
        if mapped == "Unknown" and len(sample_unknown) < 20:
            sample_unknown.append((rid, dr))
        updates.append((mapped, rid))

    # apply updates in batches
    BATCH = 1000
    for i in range(0, len(updates), BATCH):
        batch = updates[i:i+BATCH]
        args_str = ",".join(cur.mogrify("(%s,%s)", (lab, rid)).decode('utf8') for lab, rid in batch)
        # Use a temporary table to perform batch update faster
        cur.execute("CREATE TEMP TABLE tmp_map(label VARCHAR(128), id BIGINT) ON COMMIT DROP;")
        cur.execute(f"INSERT INTO tmp_map(label, id) VALUES {args_str};")
        cur.execute(f"UPDATE {TABLE} t SET cause_label = tm.label FROM tmp_map tm WHERE t.id = tm.id;")
        conn.commit()

    print("Counts after mapping:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    if sample_unknown:
        print("\nSample Unknown reason rows (id, delay_reason):")
        for rid, dr in sample_unknown:
            print(rid, "->", dr)

    cur.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
