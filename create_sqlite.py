import pandas as pd
import sqlite3

# ===== CONFIGURATION =====
CSV_FILE = "data_mnist.csv"        # Path to MNIST CSV file
DB_FILE = "SQLITE/mnist"          # Output SQLite database file
TABLE_NAME = "mnist"          # Table name
BATCH_SIZE = 1000             # Batch size for insert

# ===== STEP 1 — Read CSV =====
print("Reading CSV file...")
df = pd.read_csv(CSV_FILE)
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# ===== STEP 2 — Create SQLite DB =====
print("Connecting to SQLite...")
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# ===== STEP 3 — Create table dynamically =====
print("Creating table...")
columns = df.columns
col_defs = ", ".join([f'"{col}" INTEGER' for col in columns])
cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
cursor.execute(f"CREATE TABLE {TABLE_NAME} ({col_defs});")
conn.commit()

# ===== STEP 4 — Insert data in batches =====
print("Inserting data...")
insert_sql = f"INSERT INTO {TABLE_NAME} ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})"

for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df.iloc[start:end].values.tolist()
    cursor.executemany(insert_sql, batch)
    conn.commit()
    print(f"Inserted rows {start}–{end}")

# ===== STEP 5 — Done =====
print("Done inserting data!")
conn.close()