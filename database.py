"""
database.py - local SQLite storage for the Smart Lighting System.

Uses only the Python standard library (sqlite3) + pandas. No server needed:
all data lives in one file, lighting.db, next to this script.

Usage
-----
    python database.py                 -> create DB and load sample data (only if empty)
    python database.py add 4 120 28    -> add one reading: occupancy=4, lux=120, temp=28 C
    python database.py show            -> print the latest readings
    python database.py reset           -> delete all rows and reload sample data

Later, your ESP32 receiver script can simply call insert_reading(...).
"""

import csv
import math
import random
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd

DB_PATH = Path(__file__).with_name("lighting.db")
RAW_CSV = Path(__file__).with_name("raw_sensor_data.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,      -- ISO format: 2026-09-30T19:00:00
    occupancy        INTEGER NOT NULL,      -- number of people
    ambient_lux      REAL    NOT NULL,      -- ambient light in lux
    temperature      REAL    NOT NULL,      -- degrees Celsius
    lighting_demand  REAL                   -- predicted/used demand 0-100 % (optional)
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON sensor_readings(timestamp);
"""

# Ready-made datasets (hour:minute, occupancy, lux, temperature). The LAST row is "now".
DATASETS = {
    "Sample day (demo)": [
        ("08:00", 1, 320, 24.0), ("10:00", 4, 520, 25.5), ("12:00", 6, 640, 27.0),
        ("14:00", 5, 600, 28.0), ("16:00", 4, 420, 28.0), ("18:00", 3, 240, 27.5),
        ("19:00", 3, 150, 27.0),
    ],
    "Busy exam day": [
        ("08:00", 2, 300, 25.0), ("10:00", 8, 480, 27.0), ("12:00", 12, 600, 29.0),
        ("14:00", 10, 560, 30.0), ("16:00", 9, 380, 30.0), ("18:00", 7, 200, 29.0),
        ("19:00", 6, 90, 28.0),
    ],
    "Quiet holiday": [
        ("08:00", 0, 320, 24.0), ("10:00", 0, 540, 25.0), ("12:00", 1, 660, 27.0),
        ("14:00", 0, 620, 28.0), ("16:00", 0, 430, 28.0), ("18:00", 0, 230, 27.0),
        ("19:00", 0, 110, 26.0),
    ],
    "Cloudy rainy day": [
        ("08:00", 1, 120, 22.0), ("10:00", 4, 160, 22.0), ("12:00", 5, 190, 23.0),
        ("14:00", 4, 170, 23.0), ("16:00", 3, 120, 23.0), ("18:00", 3, 70, 22.0),
        ("19:00", 2, 25, 22.0),
    ],
}
DEFAULT_DATASET = "Sample day (demo)"
SAMPLE_ROWS = DATASETS[DEFAULT_DATASET]


def _connect():
    return closing(sqlite3.connect(DB_PATH))


def init_db() -> None:
    """Create the table if it doesn't exist yet."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def insert_reading(occupancy, ambient_lux, temperature,
                   lighting_demand=None, timestamp=None) -> None:
    """Store one sensor reading. timestamp defaults to 'now'."""
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sensor_readings "
            "(timestamp, occupancy, ambient_lux, temperature, lighting_demand) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, int(occupancy), float(ambient_lux), float(temperature),
             None if lighting_demand is None else float(lighting_demand)),
        )
        conn.commit()


def count_readings() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]


def get_latest_reading():
    """Most recent reading as a dict, or None if the table is empty."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_history(limit: int = 50) -> pd.DataFrame:
    """Latest `limit` readings, oldest first (ready for plotting)."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?",
            conn, params=(limit,),
        )
    return df.iloc[::-1].reset_index(drop=True)


def clear_readings() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM sensor_readings")
        conn.commit()


def generate_raw_csv(path: Path = RAW_CSV, seed: int = 7) -> None:
    """
    Create a realistic RAW sensor log (06:00-23:45, every 15 min) with noise and a few
    sensor faults (dropout, spike, -127 error), like a real ESP32 log would have.
    Replace raw_sensor_data.csv with your own log any time (same 4 columns).
    """
    rng = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "occupancy", "ambient_lux", "temperature"])
        for step in range(72):
            minutes = 6 * 60 + step * 15
            h = minutes / 60
            hh, mm = divmod(minutes, 60)
            occ_base = 7 * math.exp(-((h - 13) ** 2) / 18) + 2.5 * math.exp(-((h - 19) ** 2) / 4)
            occ = max(0, round(rng.gauss(occ_base, 0.8)))
            lux = max(0.0, 650 * max(0.0, math.sin((h - 6) / 12 * math.pi)) + rng.gauss(0, 25))
            temp = 24 + 5 * math.sin((h - 9) / 24 * 2 * math.pi) + rng.gauss(0, 0.3)
            lux_s, temp_s = f"{lux:.1f}", f"{temp:.2f}"
            if step in (17, 41):
                lux_s = ""          # sensor dropout
            if step == 29:
                lux_s = "9999"      # spike
            if step == 53:
                temp_s = "-127"     # typical temperature-sensor error value
            w.writerow([f"{hh:02d}:{mm:02d}", occ, lux_s, temp_s])


def ensure_raw_csv() -> None:
    if not RAW_CSV.exists():
        generate_raw_csv()


def load_dataset(name: str = DEFAULT_DATASET) -> None:
    """Replace ALL rows with one of the ready-made datasets (dated today)."""
    init_db()
    today = datetime.now().strftime("%Y-%m-%d")
    with _connect() as conn:
        conn.execute("DELETE FROM sensor_readings")
        conn.commit()
    for hhmm, occ, lux, temp in DATASETS[name]:
        insert_reading(occ, lux, temp, timestamp=f"{today}T{hhmm}:00")


def seed_sample_data() -> bool:
    """Fill an EMPTY database with the demo day. Returns True if data was added."""
    init_db()
    if count_readings() > 0:
        return False
    load_dataset(DEFAULT_DATASET)
    return True


def reset_db() -> None:
    load_dataset(DEFAULT_DATASET)


if __name__ == "__main__":
    init_db()
    args = sys.argv[1:]
    if not args:
        print("Sample data loaded." if seed_sample_data() else "Database already has data.")
        print(f"Database file: {DB_PATH}  ({count_readings()} rows)")
    elif args[0] == "add" and len(args) == 4:
        insert_reading(args[1], args[2], args[3])
        print("Reading added:", get_latest_reading())
    elif args[0] == "show":
        print(get_history(15).to_string(index=False))
    elif args[0] == "rawdata":
        generate_raw_csv()
        print("Raw data written to", RAW_CSV)
    elif args[0] == "reset":
        reset_db()
        print("Database reset and sample data reloaded.")
    else:
        print(__doc__)
