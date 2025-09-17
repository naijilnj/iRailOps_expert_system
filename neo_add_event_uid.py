#   
"""
Add a stable event_uid to TrainEvent nodes for reliable matching.

event_uid format:
  {train_no}_{station_code}_{seq}_{scheduled_arrival}

Idempotent: skips nodes that already have event_uid.
"""

from neo4j import GraphDatabase, basic_auth
import os
import math

NEO_URI = os.getenv("NEO_URI", "bolt://127.0.0.1:7687")
NEO_USER = os.getenv("NEO_USER", "neo4j")
NEO_PASS = os.getenv("NEO_PASS", "railway123")

BATCH = 500

driver = GraphDatabase.driver(NEO_URI, auth=basic_auth(NEO_USER, NEO_PASS), max_connection_lifetime=1000)

def fetch_batch(session, skip, limit):
    q = """
    MATCH (e:TrainEvent)
    WHERE e.event_uid IS NULL
    RETURN elementId(e) AS elemId, e.train_no AS train_no, e.station_code AS station_code, e.seq AS seq, e.scheduled_arrival AS sched
    SKIP $skip LIMIT $limit
    """
    res = session.run(q, skip=skip, limit=limit)
    return [ {"elemId": r["elemId"], "train_no": r["train_no"], "station_code": r["station_code"], "seq": r["seq"], "sched": r["sched"]} for r in res ]

def update_batch(session, rows):
    if not rows:
        return
    q = """
    UNWIND $rows AS r
    MATCH (e) WHERE elementId(e) = r.elemId
    SET e.event_uid = r.uid
    """
    session.run(q, rows=rows)

def make_uid(train_no, station_code, seq, sched):
    if train_no is None:
        train_no = "UNKTRAIN"
    if station_code is None:
        station_code = "UNKST"
    seq_s = str(int(seq)) if seq is not None else "0"
    sched_s = str(sched).replace(":", "-") if sched is not None else "NOSCHED"
    # sanitize whitespace
    return f"{train_no}_{station_code}_{seq_s}_{sched_s}"

def main():
    print("Connecting to Neo4j:", NEO_URI)
    updated_total = 0
    with driver.session() as sess:
        # quick count
        cnt_q = "MATCH (e:TrainEvent) WHERE e.event_uid IS NULL RETURN count(e) AS c"
        try:
            total = sess.run(cnt_q).single()["c"]
        except Exception as ex:
            print("Count query failed:", ex)
            return
        print("TrainEvent nodes missing event_uid:", total)
        if total == 0:
            print("Nothing to do.")
            return

        skip = 0
        while True:
            batch = fetch_batch(sess, skip, BATCH)
            if not batch:
                break
            rows_to_upd = []
            for r in batch:
                uid = make_uid(r.get("train_no"), r.get("station_code"), r.get("seq"), r.get("sched"))
                rows_to_upd.append({"elemId": r["elemId"], "uid": uid})
            if rows_to_upd:
                try:
                    update_batch(sess, rows_to_upd)
                    updated_total += len(rows_to_upd)
                    print(f"Updated {len(rows_to_upd)} nodes (total updated: {updated_total})")
                except Exception as e:
                    print("Batch update failed, trying single updates...", e)
                    for ru in rows_to_upd:
                        try:
                            sess.run("MATCH (e) WHERE elementId(e) = $eid SET e.event_uid = $uid", eid=ru["elemId"], uid=ru["uid"])
                            updated_total += 1
                        except Exception as e2:
                            print("Failed single update for", ru["elemId"], e2)
            skip += BATCH
    print("Done. total updated:", updated_total)
    driver.close()

if __name__ == "__main__":
    main()
