# test_neo_connect.py
from neo4j import GraphDatabase, basic_auth
import os, sys

URI = os.getenv("NEO_URI", "bolt://127.0.0.1:7687")   # try bolt:// first
USER = os.getenv("NEO_USER", "neo4j")
PWD = os.getenv("NEO_PASS", "railway123")

print("Trying URI:", URI, "user:", USER)
try:
    driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PWD))
    with driver.session() as s:
        r = s.run("RETURN 1 AS v").single()
        print("Connected OK, test query result:", r["v"])
    driver.close()
except Exception as e:
    print("Connection failed:", e)
    sys.exit(1)
