import sqlite3
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

conn = sqlite3.connect("data/hycosys_tables.db")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print(f"Found {len(tables)} tables:\n")
for t in tables:
    df = pd.read_sql(f'SELECT * FROM "{t}"', conn)
    print("=" * 70)
    print(f"{t}   shape={df.shape}")
    print("=" * 70)
    print(df.to_string(index=False))
    print()