# app_api.py
"""
Patched Railway Expert System API
- Expects delay_xgb.joblib and encoders.joblib (or legacy names)
- Decodes ML integer predictions back to class strings using saved LabelEncoder
- Keeps rules + Neo4j integration + simulation endpoints
"""

from fastapi import FastAPI, HTTPException
from typing import List, Dict
from datetime import datetime, date
import joblib
import psycopg2
import pandas as pd
import traceback
import json
import os

# Neo4j driver
from neo4j import GraphDatabase, basic_auth

# ---------- CONFIG ----------
DB_NAME = os.getenv("DB_NAME", "railway")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "142003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Model/encoders (prefer the freshly trained names)
MODEL_FILE_CANDIDATES = ["delay_xgb.joblib", "delay_classifier.joblib", "delay_model.joblib"]
ENC_FILE_CANDIDATES = ["encoders.joblib", "cat_mappings.joblib", "encoders_v1.joblib"]

# Neo4j config
NEO_URI = os.getenv("NEO_URI", "bolt://127.0.0.1:7687")
NEO_USER = os.getenv("NEO_USER", "neo4j")
NEO_PASS = os.getenv("NEO_PASS", "railway123")
# ----------------------------

app = FastAPI(title="Railway Expert System API 🚆 (patched)")

# ---------- Load model + encoders ----------
def find_existing(fname_list):
    for f in fname_list:
        if os.path.exists(f):
            return f
    return None

MODEL_FILE = find_existing(MODEL_FILE_CANDIDATES)
ENC_FILE = find_existing(ENC_FILE_CANDIDATES)

model = None
encoders_store = None
label_encoder = None
categorical_encoders = {}

if MODEL_FILE:
    try:
        model = joblib.load(MODEL_FILE)
        print(f"Loaded model from {MODEL_FILE}")
    except Exception as e:
        print(f"Warning: could not load model {MODEL_FILE}: {e}")
        model = None
else:
    print("No model file found (expected one of):", MODEL_FILE_CANDIDATES)

if ENC_FILE:
    try:
        encoders_store = joblib.load(ENC_FILE)
        if isinstance(encoders_store, dict):
            categorical_encoders = encoders_store.get("categorical_encoders", {})
            label_encoder = encoders_store.get("label_encoder", None)
        else:
            # If older format: try to find nested encoders
            label_encoder = encoders_store.get("label_encoder") if hasattr(encoders_store, "get") else None
            categorical_encoders = encoders_store.get("categorical_encoders", {}) if hasattr(encoders_store, "get") else {}
        print(f"Loaded encoders from {ENC_FILE}. Label encoder present: {label_encoder is not None}")
    except Exception as e:
        print(f"Warning: could not load encoders {ENC_FILE}: {e}")
        encoders_store = None
else:
    print("No encoders file found (expected one of):", ENC_FILE_CANDIDATES)

# ---------------- Helpers ----------------
def _safe_time_str(t):
    """Normalize a time string to HH:MM:SS or return None."""
    if not t:
        return None
    if isinstance(t, (float, int)) and pd.isna(t):
        return None
    s = str(t).strip()
    if s.lower() in {"none", "nan", ""}:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            h, m = parts
            return f"{int(h):02d}:{int(m):02d}:00"
        elif len(parts) == 3:
            h, m, sec = parts
            return f"{int(h):02d}:{int(m):02d}:{int(sec):02d}"
    except Exception:
        return None
    return None

def encode_cat(col_name, val):
    """Encode categorical using saved LabelEncoder; return fallback int if unknown."""
    if val is None:
        return -1
    le = categorical_encoders.get(col_name)
    if le is None:
        # try simple string -> hash fallback
        try:
            return abs(hash(str(val))) % 100000
        except Exception:
            return -1
    try:
        return int(le.transform([str(val)])[0])
    except Exception:
        # unknown category -> try to add fallback by using -1 or nearest index 0
        return -1

def decode_label(pred):
    """Decode integer label to string using label_encoder if present."""
    if pred is None:
        return None
    if label_encoder is None:
        # fallback: if model predicted strings already, return raw
        return str(pred)
    try:
        # pred may be array-like or scalar
        if hasattr(pred, "__len__") and not isinstance(pred, (str, bytes)):
            p0 = int(pred[0])
            return label_encoder.inverse_transform([p0])[0]
        else:
            return label_encoder.inverse_transform([int(pred)])[0]
    except Exception:
        try:
            return str(pred)
        except Exception:
            return None

# ---------------- Rules Engine ----------------
def apply_rules(train_no, station_code, delay_minutes, conn):
    """
    Return (reason_label, explanation_text) if a rule applies, else (None, None).
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT arrival_time, seq
            FROM train_timetable
            WHERE train_no = %s AND station_code = %s
            LIMIT 1
        """, (train_no, station_code))
        row = cur.fetchone()
        if not row:
            return None, None
        scheduled_time_raw, seq = row
        scheduled_time = _safe_time_str(scheduled_time_raw)

        # Rule 1: Congestion — other trains within ±10 minutes at same station
        if scheduled_time:
            cur.execute("""
                SELECT arrival_time FROM train_timetable
                WHERE station_code = %s AND train_no != %s AND arrival_time IS NOT NULL
            """, (station_code, train_no))
            others = cur.fetchall()
            for (other_time_raw,) in others:
                other_time = _safe_time_str(other_time_raw)
                if other_time:
                    s_dt = datetime.combine(date.today(), datetime.strptime(scheduled_time, "%H:%M:%S").time())
                    o_dt = datetime.combine(date.today(), datetime.strptime(other_time, "%H:%M:%S").time())
                    diff = abs((s_dt - o_dt).total_seconds())
                    if diff <= 600:
                        return "Congestion", f"Multiple trains scheduled at {station_code} around {scheduled_time} (±10 min)."

        # Rule 2: Infrastructure — large delay at small station
        cur.execute("SELECT COUNT(DISTINCT train_no) FROM train_timetable WHERE station_code = %s", (station_code,))
        station_count_row = cur.fetchone()
        station_count = station_count_row[0] if station_count_row and station_count_row[0] is not None else 0
        if delay_minutes > 30 and station_count < 5:
            return "Infrastructure", f"Station {station_code} has only {station_count} trains; long delay (>30m) suggests infrastructure issue."

        # Rule 3: Operational — origin delays
        if seq is not None and seq == 1 and delay_minutes > 10:
            return "Operational", f"Train started late at origin (seq=1) by {delay_minutes} minutes."

        return None, None
    except Exception:
        traceback.print_exc()
        return None, None

# ---------------- Recovery heuristics ----------------
def estimate_impact_skip_halt(delay_minutes: int) -> int:
    return min(max(int(delay_minutes * 0.3), 4), 15)

def count_platform_conflicts(station_code: str, scheduled_time_str: str, conn) -> int:
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM train_timetable
        WHERE station_code = %s
          AND arrival_time IS NOT NULL
          AND arrival_time::time BETWEEN (%s::time - interval '10 minutes') AND (%s::time + interval '10 minutes')
    """, (station_code, scheduled_time_str, scheduled_time_str))
    return cur.fetchone()[0]

# ---------------- Neo4j integration (lazy) ----------------
neo_driver = None

def get_neo_driver():
    """Lazily create and return a Neo4j driver. Safe to call repeatedly."""
    global neo_driver
    if neo_driver is not None:
        return neo_driver
    try:
        uri = os.getenv("NEO_URI", NEO_URI)
        user = os.getenv("NEO_USER", NEO_USER)
        pwd = os.getenv("NEO_PASS", NEO_PASS)
        neo_driver = GraphDatabase.driver(uri, auth=basic_auth(user, pwd))
        with neo_driver.session() as s:
            s.run("RETURN 1").single()
        print(f"Neo4j driver initialized to {uri}")
        return neo_driver
    except Exception as e:
        print("Could not initialize Neo4j driver:", e)
        neo_driver = None
        return None

def _neo_run_single(query: str, params: dict = None, retry: bool = True):
    params = params or {}
    d = get_neo_driver()
    if d is None:
        return None
    try:
        with d.session() as sess:
            res = sess.run(query, **params)
            return res.single()
    except Exception as ex:
        print("Neo4j query error:", ex)
        if retry:
            try:
                if neo_driver is not None:
                    try:
                        neo_driver.close()
                    except Exception:
                        pass
                globals()['neo_driver'] = None
            except Exception:
                pass
            d2 = get_neo_driver()
            if d2:
                try:
                    with d2.session() as sess:
                        res = sess.run(query, **params)
                        return res.single()
                except Exception as ex2:
                    print("Neo4j retry failed:", ex2)
                    return None
        return None

def is_feasible_overtake(train_no: str, station_code: str, conn) -> (bool, str):
    cypher = """
    MATCH (s:Station {code:$station})
    OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:TrainEvent)
      WHERE e.train_no <> $train_no AND e.scheduled_arrival IS NOT NULL
    WITH s, count(e) AS nearby
    RETURN nearby AS nearby, COUNT((s)--()) AS degree
    """
    rec = _neo_run_single(cypher, {"station": station_code, "train_no": train_no})
    if rec is None:
        return False, "Neo4j unavailable — overtaking feasibility unknown (fallback: not feasible)."
    nearby = rec.get("nearby", 0) or 0
    degree = rec.get("degree", 0) or 0
    if nearby >= 1 and degree >= 2:
        return True, f"Overtake feasible heuristic: {nearby} nearby scheduled event(s); station connectivity degree={degree}."
    if nearby >= 1 and degree < 2:
        return False, f"Nearby events found ({nearby}) but station connectivity low (degree={degree}); require manual check for loops."
    return False, "No nearby events found within station schedule — overtaking not feasible by conservative check."

def is_feasible_platform_swap(train_no: str, station_code: str, conn) -> (bool, str):
    cypher_platforms = """
    MATCH (s:Station {code:$station})
    OPTIONAL MATCH (s)-[:HAS_PLATFORM]->(p:Platform)
    RETURN count(p) AS platform_count, COUNT((s)--()) AS degree
    """
    rec = _neo_run_single(cypher_platforms, {"station": station_code})
    if rec is None:
        return False, "Neo4j unavailable — platform-swap feasibility unknown (fallback: not feasible)."
    platform_count = rec.get("platform_count", 0) or 0
    degree = rec.get("degree", 0) or 0
    if platform_count >= 2:
        return True, f"Platform swap feasible: station has {platform_count} platforms."
    if degree >= 2:
        return True, f"Platform swap heuristic OK (no explicit platform nodes); station connectivity degree={degree} suggests spare track capacity."
    return False, f"Platform swap not feasible: platform_count={platform_count}, degree={degree}."

# ---------------- API endpoints ----------------
@app.get("/")
def root():
    return {"status": "Railway Expert System API running 🚆 (patched)"}

@app.get("/neo_health")
def neo_health():
    d = get_neo_driver()
    if d is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized or connection failed")
    try:
        rec = _neo_run_single("MATCH (n) RETURN count(n) AS cnt")
        cnt = rec["cnt"] if rec else None
        return {"neo_connected": True, "node_count": cnt}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/delay_explanation")
def delay_explanation(train_no: str, station_code: str, delay_minutes: int):
    """
    Returns rule-based reason (if applicable) + ML prediction fallback.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)

        # 1) rules
        rule_reason, rule_text = apply_rules(train_no, station_code, delay_minutes, conn)

        # 2) prepare ML features
        cur = conn.cursor()
        cur.execute("""
            SELECT seq, COALESCE(distance,0), source_station, destination_station
            FROM train_timetable
            WHERE train_no = %s AND station_code = %s
            LIMIT 1
        """, (train_no, station_code))
        meta = cur.fetchone()
        if not meta:
            conn.close()
            raise HTTPException(status_code=404, detail="train/station not found in timetable")
        seq, distance, source_station, destination_station = meta

        # encode categories using saved encoders
        train_code = encode_cat("train_no", train_no)
        station_code_enc = encode_cat("station_code", station_code)
        source_code = encode_cat("source_station", source_station)
        dest_code = encode_cat("destination_station", destination_station)

        X = pd.DataFrame([{
            "train_no_code": train_code,
            "station_code_code": station_code_enc,
            "seq": int(seq) if seq is not None else -1,
            "distance": int(distance) if distance is not None else 0,
            "delay_minutes": int(delay_minutes),
            "source_station_code": source_code,
            "destination_station_code": dest_code
        }])

        ml_pred = None
        ml_pred_str = None
        if model is not None:
            try:
                # Model expects numeric columns named as used during training:
                # training used ["train_no_enc","station_code_enc","seq","distance","delay_minutes"]
                X_for_model = X.rename(columns={"train_no_code": "train_no_enc", "station_code_code": "station_code_enc"})
                # ensure column order
                X_for_model = X_for_model[["train_no_enc", "station_code_enc", "seq", "distance", "delay_minutes"]]
                pred = model.predict(X_for_model)
                ml_pred = pred[0] if hasattr(pred, "__len__") else pred
                ml_pred_str = decode_label(ml_pred)
            except Exception:
                traceback.print_exc()
                ml_pred = None
                ml_pred_str = None

        conn.close()

        final_reason = rule_reason if rule_reason is not None else (ml_pred_str if ml_pred_str is not None else "Unknown")

        result = {
            "train_no": train_no,
            "station_code": station_code,
            "delay_minutes": delay_minutes,
            "rule_based_reason": rule_reason,
            "rule_explanation": rule_text,
            "ml_predicted_reason": ml_pred_str,
            "final_reason": final_reason
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# recovery_recommendation, simulate_action, simulate_compare, approve_recommendation
# ---------------------------------------------------------
# For brevity include the existing functions with no change in logic — reuse those
# (copy them exactly from your earlier app_api; below I re-include them adapted slightly)
# ---------------------------------------------------------

@app.get("/recovery_recommendation")
def recovery_recommendation(train_no: str, station_code: str, delay_minutes: int):
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()

        cur.execute("""
            SELECT arrival_time, departure_time, seq
            FROM train_timetable
            WHERE train_no = %s AND station_code = %s
            LIMIT 1
        """, (train_no, station_code))
        meta = cur.fetchone()
        if not meta:
            conn.close()
            raise HTTPException(status_code=404, detail="train/station not found in timetable")

        scheduled_arrival_raw, scheduled_departure_raw, seq = meta
        scheduled_arrival_str = _safe_time_str(scheduled_arrival_raw)
        scheduled_departure_str = _safe_time_str(scheduled_departure_raw)

        candidates: List[Dict] = []

        if delay_minutes < 5:
            candidates.append({
                "action": "no_action",
                "impact_estimate_min_saved": 0,
                "reason": f"Delay < 5 minutes ({delay_minutes}) — no intervention recommended."
            })
            conn.close()
            return {"train_no": train_no, "station_code": station_code, "delay_minutes": delay_minutes, "candidates": candidates}

        conflicts = 0
        if scheduled_arrival_str is not None:
            try:
                conflicts = count_platform_conflicts(station_code, scheduled_arrival_str, conn)
            except Exception:
                conflicts = 0
        if conflicts > 2:
            est = min(int(delay_minutes * 0.4), 20)
            candidates.append({
                "action": "platform_swap",
                "impact_estimate_min_saved": est,
                "reason": f"{conflicts} trains scheduled near this time; platform swap may reduce dwell conflicts and recover time.",
                "risk": "Requires platform availability and staff coordination."
            })

        try:
            seq_int = int(seq) if seq is not None else -1
        except Exception:
            seq_int = -1
        if seq_int >= 3:
            est_skip = estimate_impact_skip_halt(delay_minutes)
            candidates.append({
                "action": "skip_halt",
                "impact_estimate_min_saved": est_skip,
                "reason": f"Station appears to be a middle halt (seq={seq_int}); skipping may save ~{est_skip} min.",
                "risk": "Passenger impact: check bookings and SOP for skip-halt permissions."
            })

        if delay_minutes >= 15 and scheduled_arrival_str is not None:
            cur.execute("""
                SELECT train_no, arrival_time FROM train_timetable
                WHERE station_code = %s AND train_no != %s
                  AND arrival_time IS NOT NULL
                  AND arrival_time::time BETWEEN %s::time AND (%s::time + interval '30 minutes')
                LIMIT 5
            """, (station_code, train_no, scheduled_arrival_str, scheduled_arrival_str))
            upcoming = [(r[0], _safe_time_str(r[1])) for r in cur.fetchall() if _safe_time_str(r[1]) is not None]
            if upcoming:
                candidates.append({
                    "action": "allow_overtake",
                    "impact_estimate_min_saved": min(int(delay_minutes * 0.6), 30),
                    "reason": f"Upcoming trains present ({len(upcoming)}). Consider higher-priority overtake if loop available.",
                    "risk": "Requires loop availability and signalling clearance; may shift delays to overtaken trains."
                })

        candidates.append({
            "action": "reschedule_next_section",
            "impact_estimate_min_saved": max(int(delay_minutes * 0.2), 1),
            "reason": "Fallback: minor timetable adjustments or temporary speed changes downstream to absorb delay.",
            "risk": "May cause minor knock-on delays elsewhere; simulate before approval."
        })

        candidates_sorted = sorted(candidates, key=lambda x: x["impact_estimate_min_saved"], reverse=True)
        conn.close()
        return {
            "train_no": train_no,
            "station_code": station_code,
            "delay_minutes": delay_minutes,
            "candidates": candidates_sorted
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulate_action")
def simulate_action(train_no: str, station_code: str, delay_minutes: int, action: str = "allow_overtake", horizon: int = 10):
    action = action.lower()
    if action not in {"allow_overtake", "skip_halt", "platform_swap", "reschedule_next_section"}:
        raise HTTPException(status_code=400, detail="invalid action")

    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()

        cur.execute("""
            SELECT seq FROM train_timetable
            WHERE train_no = %s AND station_code = %s
            LIMIT 1
        """, (train_no, station_code))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="train/station not found in timetable")
        start_seq = int(row[0])

        cur.execute("""
            SELECT seq, station_code, arrival_time
            FROM train_timetable
            WHERE train_no = %s AND seq >= %s
            ORDER BY seq
            LIMIT %s
        """, (train_no, start_seq, horizon))
        stops = cur.fetchall()

        if action == "allow_overtake":
            saved = int(min(delay_minutes * 0.6, 30))
        elif action == "skip_halt":
            saved = estimate_impact_skip_halt(delay_minutes)
        elif action == "platform_swap":
            saved = int(min(delay_minutes * 0.4, 20))
        else:
            saved = max(int(delay_minutes * 0.2), 1)

        adjusted_stops = []
        for seq, st_code, arr_time_raw in stops:
            sched_str = _safe_time_str(arr_time_raw)
            if sched_str is None:
                adjusted_stops.append({"seq": seq, "station_code": st_code, "scheduled_arrival": None, "adjusted_delay": None, "adjusted_arrival": None})
                continue
            sched_dt = datetime.combine(date.today(), datetime.strptime(sched_str, "%H:%M:%S").time())
            orig_delay = delay_minutes
            new_delay = max(orig_delay - saved, 0)
            adjusted_arrival = (sched_dt + pd.Timedelta(minutes=new_delay)).time().isoformat()
            adjusted_stops.append({
                "seq": int(seq),
                "station_code": st_code,
                "scheduled_arrival": sched_str,
                "original_delay_min": int(orig_delay),
                "adjusted_delay_min": int(new_delay),
                "adjusted_arrival_time": adjusted_arrival
            })

        impacted_trains = []
        for stop in adjusted_stops:
            if stop["adjusted_arrival_time"] is None:
                continue
            st = stop["station_code"]
            adjusted_time = stop["adjusted_arrival_time"]
            cur.execute("""
                SELECT train_no, arrival_time
                FROM train_timetable
                WHERE station_code = %s AND train_no != %s AND arrival_time IS NOT NULL
            """, (st, train_no))
            others = cur.fetchall()
            for other_no, other_time_raw in others:
                other_time = _safe_time_str(other_time_raw)
                if other_time is None:
                    continue
                try:
                    adj_dt = datetime.combine(date.today(), datetime.strptime(adjusted_time, "%H:%M:%S").time())
                    oth_dt = datetime.combine(date.today(), datetime.strptime(other_time, "%H:%M:%S").time())
                    diff_min = abs((adj_dt - oth_dt).total_seconds()) / 60.0
                    if diff_min <= max(5, saved):
                        added = int(max(1, saved * 0.5))
                        impacted_trains.append({
                            "station_code": st,
                            "other_train_no": other_no,
                            "other_scheduled_arrival": other_time,
                            "minutes_from_adjusted": int(diff_min),
                            "estimated_added_delay_min": added,
                            "reason": f"Adjusted arrival close to {other_no} (Δ={int(diff_min)}min); estimated +{added} min knock-on"
                        })
                except Exception:
                    continue

        conn.close()

        result = {
            "train_no": train_no,
            "starting_station": station_code,
            "action": action,
            "saved_minutes_applied": saved,
            "adjusted_downstream_stops": adjusted_stops,
            "impacted_other_trains": impacted_trains
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulate_compare")
def simulate_compare(train_no: str, station_code: str, delay_minutes: int, horizon: int = 10):
    actions = ["allow_overtake", "skip_halt", "platform_swap", "reschedule_next_section"]
    results = []
    try:
        for act in actions:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
            feasible = True
            feas_text = None
            if act == "allow_overtake":
                feasible, feas_text = is_feasible_overtake(train_no, station_code, conn)
            elif act == "platform_swap":
                feasible, feas_text = is_feasible_platform_swap(train_no, station_code, conn)
            conn.close()
            if not feasible:
                results.append({
                    "action": act,
                    "feasible": False,
                    "feasibility_note": feas_text,
                    "simulation": None,
                    "net_minutes": None,
                    "passenger_weighted_net": None
                })
                continue

            sim = simulate_action(train_no, station_code, delay_minutes, action=act, horizon=horizon)
            saved = sim.get("saved_minutes_applied", 0)
            added = sum(item.get("estimated_added_delay_min", 0) for item in sim.get("impacted_other_trains", []))
            net = -saved + added
            passenger_weighted_net = None
            try:
                conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
                cur = conn.cursor()
                cur.execute("SELECT passenger_count FROM passenger_counts WHERE train_no=%s AND station_code=%s ORDER BY snapshot_date DESC LIMIT 1", (train_no, station_code))
                r = cur.fetchone()
                main_pass = r[0] if r else None
                total_passenger_minutes = 0
                total_passengers = 0
                for it in sim.get("impacted_other_trains", []):
                    oth = it["other_train_no"]
                    st = it["station_code"]
                    cur.execute("SELECT passenger_count FROM passenger_counts WHERE train_no=%s AND station_code=%s ORDER BY snapshot_date DESC LIMIT 1", (oth, st))
                    rr = cur.fetchone()
                    oth_pass = rr[0] if rr else None
                    minutes = it.get("estimated_added_delay_min", 0)
                    if oth_pass:
                        total_passenger_minutes += oth_pass * minutes
                        total_passengers += oth_pass
                if main_pass:
                    main_saved_pass_min = main_pass * saved
                    passenger_weighted_net = total_passenger_minutes - main_saved_pass_min
                conn.close()
            except Exception:
                passenger_weighted_net = None

            results.append({
                "action": act,
                "feasible": True,
                "feasibility_note": feas_text,
                "simulation": sim,
                "net_minutes": net,
                "passenger_weighted_net": passenger_weighted_net
            })

        valid_by_net = [r for r in results if r.get("net_minutes") is not None]
        best_net = min(valid_by_net, key=lambda x: x["net_minutes"]) if valid_by_net else None
        valid_by_pw = [r for r in results if r.get("passenger_weighted_net") is not None]
        best_passenger = min(valid_by_pw, key=lambda x: x["passenger_weighted_net"]) if valid_by_pw else None

        return {
            "train_no": train_no,
            "station_code": station_code,
            "delay_minutes": delay_minutes,
            "horizon": horizon,
            "comparisons": results,
            "best_by_net_minutes": best_net,
            "best_by_passenger_weighted": best_passenger
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approve_recommendation")
def approve_recommendation(train_no: str, station_code: str, delay_minutes: int, action: str, operator: str, operator_decision: str, notes: str = None, simulation_json: dict = None):
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO recommendations_log
              (train_no, station_code, delay_minutes, action, simulation_result, operator, operator_decision, decision_ts, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            RETURNING id
        """, (train_no, station_code, delay_minutes, action, json.dumps(simulation_json) if simulation_json else None, operator, operator_decision, notes))
        new_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        return {"status": "ok", "log_id": new_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Close Neo4j driver on shutdown
@app.on_event("shutdown")
def _close_neo4j_driver():
    global neo_driver
    try:
        if neo_driver is not None:
            neo_driver.close()
            neo_driver = None
            print("Neo4j driver closed.")
    except Exception:
        pass
