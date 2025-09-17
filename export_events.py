# export_events.py
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:142003@localhost:5432/railway")
df = pd.read_sql("SELECT train_no, station_code, station_name, seq, arrival_time, departure_time FROM train_timetable", engine)
df.to_csv("train_events.csv", index=False)
print("wrote train_events.csv:", len(df))
