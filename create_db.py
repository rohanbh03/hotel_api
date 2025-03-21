import sqlite3
import pandas as pd

# Paths
DB_PATH = "hotel_bookings.db"
CSV_PATH = "cleaned_hotel_bookings.csv"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create 'bookings' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT,
    reservation_status_date TEXT,
    stays_in_weekend_nights INTEGER,
    stays_in_week_nights INTEGER,
    lead_time INTEGER,
    market_segment TEXT,
    adr REAL,
    is_canceled INTEGER
)
''')

# Create 'query_history' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    response TEXT,
    timestamp TEXT
)
''')

# Load CSV and insert into database
data = pd.read_csv(CSV_PATH)
data.to_sql("bookings", conn, if_exists="replace", index=False)

print("[INFO] Database successfully created!")

conn.commit()
conn.close()
