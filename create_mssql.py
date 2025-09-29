import pandas as pd
import pyodbc

# ===== CONFIGURATION =====
CSV_FILE = "data_mnist.csv"       # Path to MNIST CSV file
TABLE_NAME = "mnist"              # Table name
BATCH_SIZE = 1000                 # Batch size for insert

# MSSQL connection string (update with your settings!)
SERVER = "localhost,1433"         # Or your server/container name + port
DATABASE = "master"               # Target database name
USERNAME = "sa"                   # SQL Server user
PASSWORD = "Chonn@m91"        # SQL Server password

# ===== STEP 1 — Read CSV =====
print("Reading CSV file...")
df = pd.read_csv(CSV_FILE)
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# ===== STEP 2 — Connect to MSSQL =====
print("Connecting to MSSQL...")
conn = pyodbc.connect(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
)
cursor = conn.cursor()

# ===== STEP 3 — Create table dynamically =====
print("Creating table...")
columns = df.columns
col_defs = ", ".join([f"[{col}] FLOAT" for col in columns])  # Use INT since MNIST pixels are 0–255
cursor.execute(f"IF OBJECT_ID('{TABLE_NAME}', 'U') IS NOT NULL DROP TABLE {TABLE_NAME};")
cursor.execute(f"CREATE TABLE {TABLE_NAME} ({col_defs});")
conn.commit()

# ===== STEP 4 — Insert data in batches =====
print("Inserting data...")
placeholders = ", ".join(["?" for _ in columns])
insert_sql = f"INSERT INTO {TABLE_NAME} ({', '.join([f'[{col}]' for col in columns])}) VALUES ({placeholders})"

for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df.iloc[start:end].values.tolist()
    cursor.executemany(insert_sql, batch)
    conn.commit()
    print(f"Inserted rows {start}–{end}")

# ===== STEP 5 — Done =====
print("Done inserting data!")
conn.close()
