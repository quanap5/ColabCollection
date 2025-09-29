import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# ===== CONFIGURATION =====
CSV_FILE = "data_mnist.csv"       # Path to MNIST CSV file
TABLE_NAME = "mnist"              # Table name
BATCH_SIZE = 1000                 # Batch size for insert

# PostgreSQL connection settings
HOST = "localhost"                # Or your container hostname
PORT = "5432"                     # Default PostgreSQL port
DATABASE = "testdb"                 # Target database name
USERNAME = "admin"                # PostgreSQL user
PASSWORD = "Chonn@m91"      # PostgreSQL password

# ===== STEP 1 — Read CSV =====
print("Reading CSV file...")
df = pd.read_csv(CSV_FILE)
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# ===== STEP 2 — Connect to PostgreSQL =====
print("Connecting to PostgreSQL...")
conn = psycopg2.connect(
    host=HOST,
    port=PORT,
    dbname=DATABASE,
    user=USERNAME,
    password=PASSWORD
)
cursor = conn.cursor()

# ===== STEP 3 — Create table dynamically =====
print("Creating table...")
columns = df.columns
# PostgreSQL identifiers don’t need square brackets
col_defs = ", ".join([f"\"{col}\" REAL" for col in columns])  # REAL = floating-point
cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
cursor.execute(f"CREATE TABLE {TABLE_NAME} ({col_defs});")
conn.commit()

# ===== STEP 4 — Insert data in batches =====
print("Inserting data...")
col_list = ", ".join([f'"{col}"' for col in columns])
placeholders = ", ".join(["%s" for _ in columns])
insert_sql = f"INSERT INTO {TABLE_NAME} ({col_list}) VALUES ({placeholders})"

for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df.iloc[start:end].values.tolist()
    execute_batch(cursor, insert_sql, batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"Inserted rows {start}–{end}")

# ===== STEP 5 — Done =====
print("Done inserting data!")
cursor.close()
conn.close()
