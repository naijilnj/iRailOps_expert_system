# ingest_topology.py
import os
import pandas as pd
from sqlalchemy import create_engine
from neo4j import GraphDatabase
from datetime import datetime
import math

# --------------- CONFIG ---------------
# Postgres
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "142003")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "railway")

# Neo4j (Bolt)neo4j://127.0.0.1:7687
NEO_URI = os.getenv("NEO_URI", "bolt://localhost:7687")
NEO_USER = os.getenv("NEO_USER", "neo4j")
NEO_PASS = os.getenv("NEO_PASS", "railway123")  # change to your password

# Optional: how many sample trains to use (None = all)
SAMPLE_TRAINS = None
# --------------------------------------

def time_to_minutes(t):
    """
    Accepts time-like strings like "12:10:00", "0:00:00", "12:10" or None.
    Returns minutes since midnight (int) or None.
    """
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    s = str(t).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            h, m = parts
            return int(h) * 60 + int(m)
        elif len(parts) == 3:
            h, m, sec = parts
            return int(h) * 60 + int(m)
    except Exception:
        return None
    return None

def extract_adjacent_pairs(df_tt):
    """
    df_tt expected columns: train_no, station_code, seq, arrival_time, departure_time, distance (optional)
    Returns DataFrame of edges with columns: from_code, to_code, travel_min, train_no
    """
    edges = []
    # operate per train
    for train_no, g in df_tt.groupby("train_no"):
        g_sorted = g.sort_values("seq")
        prev = None
        for _, row in g_sorted.iterrows():
            station = row["station_code"]
            seq = row["seq"]
            arr = row.get("arrival_time")
            arr_min = time_to_minutes(arr)
            dist = row.get("distance", None)
            if prev is not None:
                # compute travel time from prev arrival to current arrival if both available
                if prev["arr_min"] is not None and arr_min is not None:
                    travel = arr_min - prev["arr_min"]
                    # handle day rollover (negative travel) assuming next-day arrival -> add 24*60
                    if travel < 0:
                        travel += 24*60
                else:
                    travel = None
                edges.append({
                    "train_no": train_no,
                    "from_code": prev["station"],
                    "to_code": station,
                    "from_seq": prev["seq"],
                    "to_seq": seq,
                    "travel_min": travel,
                    "distance_from": prev.get("distance"),
                    "distance_to": dist
                })
            prev = {"station": station, "seq": seq, "arr_min": arr_min, "distance": dist}
    return pd.DataFrame(edges)

def aggregate_edges(df_edges):
    """
    Aggregate edges by (from_code,to_code) and compute average travel_min and sample_count.
    Returns DataFrame with columns: from_code, to_code, avg_travel_min, sample_count, avg_distance_delta
    """
    # compute distance delta where available
    def compute_delta(grp):
        # drop None travel_min
        travel_list = [v for v in grp["travel_min"].tolist() if v is not None]
        avg_travel = sum(travel_list)/len(travel_list) if travel_list else None
        # distance delta if available
        dist_deltas = []
        for a,b in zip(grp.get("distance_from", []), grp.get("distance_to", [])):
            try:
                if a is not None and b is not None:
                    dist_deltas.append(float(b) - float(a))
            except Exception:
                continue
        avg_dist = sum(dist_deltas)/len(dist_deltas) if dist_deltas else None
        return pd.Series({
            "avg_travel_min": avg_travel,
            "sample_count": len(grp),
            "avg_distance_delta": avg_dist
        })
    agg = df_edges.groupby(["from_code","to_code"]).apply(compute_delta).reset_index()
    return agg

def create_graph(agg_edges, neo_uri, neo_user, neo_pass):
    driver = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pass))
    with driver.session() as session:
        # Create uniqueness constraint for station code
        session.run("CREATE CONSTRAINT station_code_unique IF NOT EXISTS FOR (s:Station) REQUIRE s.code IS UNIQUE;")
        # create nodes
        stations = set(agg_edges["from_code"].tolist()) | set(agg_edges["to_code"].tolist())
        print(f"Creating/merging {len(stations)} Station nodes...")
        for code in stations:
            session.run("MERGE (s:Station {code:$code}) SET s.name = coalesce(s.name,$code)", code=code)

        print("Creating/merging BLOCK relationships...")
        # create relationships with properties
        for _, row in agg_edges.iterrows():
            from_c = row["from_code"]
            to_c = row["to_code"]
            avg_travel = row["avg_travel_min"]
            sample_count = int(row["sample_count"]) if not pd.isna(row["sample_count"]) else 0
            avg_dist = row["avg_distance_delta"] if "avg_distance_delta" in row else None
            # relationship id: from-to
            cy = """
            MATCH (a:Station {code:$from_c}), (b:Station {code:$to_c})
            MERGE (a)-[r:BLOCK]->(b)
            SET r.avg_travel_min = $avg_travel, r.sample_count = $sample_count, r.avg_distance_delta = $avg_dist
            """
            session.run(cy, from_c=from_c, to_c=to_c, avg_travel=avg_travel, sample_count=sample_count, avg_dist=avg_dist)
    driver.close()
    print("Done ingesting graph.")

def main():
    # 1) Read timetable from Postgres
    pg_url = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(pg_url)
    # We expect train_timetable to have columns: train_no (or train_id), station_code, seq, arrival_time, departure_time, distance
    df_tt = pd.read_sql("SELECT train_no, station_code, seq, arrival_time, departure_time, distance FROM train_timetable", engine)
    if SAMPLE_TRAINS:
        df_tt = df_tt[df_tt["train_no"].isin(SAMPLE_TRAINS)]
    print("Loaded timetable rows:", len(df_tt))

    # 2) Extract adjacent pairs
    df_edges = extract_adjacent_pairs(df_tt)
    print("Extracted edge rows (per train adjacencies):", len(df_edges))

    # drop self-loops if any
    df_edges = df_edges[df_edges["from_code"] != df_edges["to_code"]]

    # 3) Aggregate by pair
    agg = aggregate_edges(df_edges)
    print("Aggregated to unique edges:", len(agg))

    # inspect a few
    print(agg.head(10).to_string(index=False))

    # 4) Push to Neo4j
    create_graph(agg, NEO_URI, NEO_USER, NEO_PASS)

if __name__ == "__main__":
    main()
