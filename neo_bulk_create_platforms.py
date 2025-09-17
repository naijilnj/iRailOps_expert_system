# neo_bulk_create_platforms.py
"""
Heuristic bulk-create of Platform nodes for busy stations.

Behavior:
- Finds stations with >= MIN_EVENTS TrainEvent nodes and with no Platform children.
- Creates N_PLATFORMS platform nodes per station and links them: (s)-[:HAS_PLATFORM]->(p)
- Marks created platform nodes with property created_by='bulk_script' so they can be removed later.

Adjust MIN_EVENTS and N_PLATFORMS as needed.
"""

from neo4j import GraphDatabase, basic_auth
import os
from math import ceil

NEO_URI = os.getenv("NEO_URI", "bolt://127.0.0.1:7687")
NEO_USER = os.getenv("NEO_USER", "neo4j")
NEO_PASS = os.getenv("NEO_PASS", "railway123")

driver = GraphDatabase.driver(NEO_URI, auth=basic_auth(NEO_USER, NEO_PASS))
MIN_EVENTS = 5      # station must have at least this many events to qualify
N_PLATFORMS = 2     # number of platforms to create per station
BATCH = 100

def find_candidate_stations(tx, min_events):
    q = """
    MATCH (s:Station)
    OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:TrainEvent)
    OPTIONAL MATCH (s)-[:HAS_PLATFORM]->(p:Platform)
    WITH s, count(e) AS events, count(p) AS platforms
    WHERE events >= $min_events AND platforms = 0
    RETURN s.code AS code, s.name AS name, events
    ORDER BY events DESC
    """
    return list(tx.run(q, min_events=min_events))

def create_platforms_for_station(tx, station_code, n):
    # create n platforms and attach to station; avoid duplicates using MERGE on platform code
    for i in range(1, n+1):
        plat_code = f"{station_code}_P{i}"
        tx.run("""
            MATCH (s:Station {code:$scode})
            MERGE (p:Platform {code:$pcode})
              ON CREATE SET p.platform_index = $idx, p.station_code = $scode, p.created_by='bulk_script'
            MERGE (s)-[:HAS_PLATFORM]->(p)
        """, scode=station_code, pcode=plat_code, idx=i)

def main():
    print("Connecting to Neo4j:", NEO_URI)
    with driver.session() as sess:
        candidates = sess.read_transaction(find_candidate_stations, MIN_EVENTS)
        print(f"Found {len(candidates)} candidate stations (events >= {MIN_EVENTS} and 0 platforms).")
        if not candidates:
            return
        # batch create
        for idx in range(0, len(candidates), BATCH):
            batch = candidates[idx: idx + BATCH]
            print(f"Processing batch {idx}..{idx+len(batch)-1}")
            with sess.begin_transaction() as tx:
                for s in batch:
                    scode = s["code"]
                    events = s["events"]
                    print(f" - Creating {N_PLATFORMS} platform(s) for {scode} (events={events})")
                    create_platforms_for_station(tx, scode, N_PLATFORMS)
                tx.commit()
        print("Done creating platforms. Verify in Neo4j Browser.")

if __name__ == "__main__":
    main()
    driver.close()
