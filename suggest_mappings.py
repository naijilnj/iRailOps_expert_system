# suggest_mappings.py
"""
Generate mapping suggestions for delay_reason -> cause_label.
Produces delay_reason_suggestions.csv for manual review.

Run:
  (venv) python suggest_mappings.py
"""

import os
import csv
import re
import psycopg2
from collections import Counter
from difflib import SequenceMatcher

DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

OUT_CSV = "delay_reason_suggestions.csv"
TOP_K = 200  # top distinct reasons to export (adjustable)

# seed label examples for fuzzy matching
EXAMPLES = {
    "Operational": [
        "crew change", "loco failure", "engine failure", "technical", "breakdown", "driver unavailable",
        "shunting", "traction", "loco attach", "loco detach", "engine failed"
    ],
    "Congestion": [
        "late incoming", "running late", "platform occupied", "traffic", "congestion", "wait due to late arrival",
        "wait for crossing", "wait for late train"
    ],
    "Infrastructure": [
        "track work", "points failure", "track failure", "permanent way", "ohle", "catenary", "bridge damage",
        "overhead equipment", "signal circuit"
    ],
    "External": [
        "fog", "rain", "storm", "accident", "trespass", "animal", "police", "road accident", "landslide", "flood"
    ]
}

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def suggest_label(text):
    if not text:
        return "Unknown"
    t = text.lower()
    # direct keyword checks (strong)
    kw_map = {
        "operational": ["crew", "loco", "engine", "driver", "technical", "breakdown", "shunting", "traction", "attach", "detach"],
        "congestion": ["late incoming", "running late", "late", "platform occupied", "platform", "congest", "traffic", "wait for"],
        "infrastructure": ["track", "point", "points", "permanent way", "perway", "ohle", "catenary", "overhead"],
        "external": ["fog", "rain", "storm", "trespass", "animal", "accident", "police", "landslide", "flood"]
    }
    for lab, kws in kw_map.items():
        for k in kws:
            if k in t:
                return lab.capitalize()

    # fuzzy compare to examples
    best_lab = "Unknown"
    best_score = 0.0
    for lab, examples in EXAMPLES.items():
        for ex in examples:
            s = similar(t, ex)
            if s > best_score:
                best_score = s
                best_lab = lab
    if best_score >= 0.6:
        return best_lab

    # fallback heuristics
    if len(t) < 25 and "delay" in t:
        return "Operational"
    return "Unknown"

def main():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        SELECT delay_reason, count(*) as cnt
        FROM train_delays
        WHERE delay_reason IS NOT NULL
        GROUP BY delay_reason
        ORDER BY cnt DESC
        LIMIT %s
    """, (TOP_K,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    rows_out = []
    for dr, cnt in rows:
        dr_strip = dr.strip() if isinstance(dr, str) else dr
        suggested = suggest_label(dr_strip)
        rows_out.append((dr_strip, cnt, suggested))

    # write CSV for manual editing
    with open(OUT_CSV, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["delay_reason", "count", "suggested_label", "final_label (edit)"])
        for r in rows_out:
            writer.writerow([r[0], r[1], r[2], ""])

    print(f"Wrote {len(rows_out)} suggestions to {OUT_CSV}")
    print("Edit the 'final_label (edit)' column to correct labels. Use exactly one of: Operational, Congestion, Infrastructure, External, Unknown")
    print("When ready, run: python apply_manual_mapping.py")

if __name__ == "__main__":
    main()
