# neo_update_arrival_min.py
"""
Update TrainEvent nodes with numeric arrival_min (minutes since midnight).

Notes:
- Uses elementId(e) (recommended) instead of deprecated id(e).
- Uses `... IS NULL` instead of NOT EXISTS(...) to be compatible with newer Cypher.
- Batch updates via UNWIND to reduce round-trips.
- Idempotent: skips TrainEvent nodes that already have arrival_min.
"""

from neo4j import GraphDatabase, basic_auth
import os
from datetime import datetime
import math

NEO_URI = os.getenv("NEO_URI", "bolt://127.0.0.1:7687")
NEO_USER = os.getenv("NEO_USER", "neo4j")
NEO_PASS = os.getenv("NEO_PASS", "railway123")

BATCH = 500

driver = GraphDatabase.driver(NEO_URI, auth=basic_auth(NEO_USER, NEO_PASS), max_connection_lifetime=1000)

def to_min(hms):
    if hms is None:
        return None
    s = str(hms).strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:
            hh, mm = parts
            ss = 0
        else:
            hh, mm, ss = parts
        hh_i = int(hh)
        mm_i = int(mm)
        ss_i = int(ss)
        return hh_i * 60 + mm_i + ss_i / 60.0
    except Exception:
        return None

def fetch_batch(session, skip, limit):
    """
    Return list of {"elemId":..., "arr":...} for TrainEvent nodes where arrival_min IS NULL
    """
    q = """
    MATCH (e:TrainEvent)
    WHERE e.scheduled_arrival IS NOT NULL AND e.arrival_min IS NULL
    RETURN elementId(e) AS elemId, e.scheduled_arrival AS arr
    SKIP $skip LIMIT $limit
    """
    res = session.run(q, skip=skip, limit=limit)
    return [ {"elemId": r["elemId"], "arr": r["arr"]} for r in res ]

def update_batch(session, rows):
    """
    rows: list of {"elemId": str, "m": float}
    """
    if not rows:
        return
    q = """
    UNWIND $rows AS r
    MATCH (e) WHERE elementId(e) = r.elemId
    SET e.arrival_min = r.m
    """
    session.run(q, rows=rows)

def main():
    print("Connecting to Neo4j:", NEO_URI)
    with driver.session() as sess:
        count_q = "MATCH (e:TrainEvent) WHERE e.scheduled_arrival IS NOT NULL AND e.arrival_min IS NULL RETURN count(e) AS c"
        try:
            count = sess.run(count_q).single()["c"]
        except Exception as ex:
            print("Count query failed:", ex)
            return
        print(f"TrainEvent nodes to update (arrival_min IS NULL): {count}")
        if count == 0:
            print("Nothing to do.")
            return

        skip = 0
        updated_total = 0
        while True:
            batch = fetch_batch(sess, skip, BATCH)
            if not batch:
                break

            rows_to_update = []
            for r in batch:
                arr = r["arr"]
                m = to_min(arr)
                if m is None or (isinstance(m, float) and math.isnan(m)):
                    continue
                rows_to_update.append({"elemId": r["elemId"], "m": float(m)})

            if rows_to_update:
                try:
                    update_batch(sess, rows_to_update)
                    updated_total += len(rows_to_update)
                    print(f"Updated {len(rows_to_update)} nodes (total updated: {updated_total})")
                except Exception as e:
                    print("Batch update failed, attempting per-node fallback:", e)
                    for r in rows_to_update:
                        try:
                            sess.run("MATCH (e) WHERE elementId(e) = $eid SET e.arrival_min = $m", eid=r["elemId"], m=r["m"])
                            updated_total += 1
                        except Exception as e2:
                            print("Single update failed for", r["elemId"], e2)
            else:
                print("No valid times in this batch to update.")

            skip += BATCH

        print("Done. total updated:", updated_total)

if __name__ == "__main__":
    main()
    driver.close()
