import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# ===== CONFIGURATION =====
CSV_FILE = "data_mnist.csv"       # Path to MNIST CSV file
TABLE_NAME = "mnist"              # Table name
BATCH_SIZE = 1000                 # Batch size for insert

# MySQL connection configuration
MYSQL_USER = "root"
MYSQL_PASSWORD = "Chonn@m91"
MYSQL_HOST = "localhost"          # Or your Docker container name
MYSQL_PORT = 3306
MYSQL_DB = "mnist"

# ===== STEP 1 — Read CSV =====
print("Reading CSV file...")
df = pd.read_csv(CSV_FILE)
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# ===== STEP 2 — Connect to MySQL =====
print("Connecting to MySQL...")
password_encoded = quote_plus(MYSQL_PASSWORD)
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{password_encoded}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

# ===== STEP 3 — Create table dynamically =====
print("Creating table...")
with engine.connect() as conn:
    col_defs = ", ".join([f"`{col}` FLOAT" for col in df.columns])  # MNIST pixels are integers
    conn.execute(text(f"DROP TABLE IF EXISTS `{TABLE_NAME}`;"))
    conn.execute(text(f"CREATE TABLE `{TABLE_NAME}` ({col_defs});"))

# ===== STEP 4 — Insert data in batches =====
print("Inserting data...")
for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_df = df.iloc[start:end]
    batch_df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False)
    print(f"Inserted rows {start}–{end}")

# ===== STEP 5 — Done =====
print("Done inserting data!")
