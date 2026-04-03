"""
database.py — SQLite database operations.

WHY SQLite?
- It's a file-based database (no server to install)
- Perfect for projects that run on a single machine
- The data persists between runs (unlike a pandas DataFrame in memory)
- Shows interviewers you understand data storage, not just notebooks

HOW IT WORKS:
- SQLAlchemy creates a 'financial_data.db' file in your data/ folder
- We write pandas DataFrames directly to SQL tables
- We read them back with SQL queries
"""

from sqlalchemy import create_engine
import pandas as pd
import os

# Create database engine — this creates the .db file if it doesn't exist
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "financial_data.db")
engine = create_engine(f"sqlite:///{DB_PATH}")


def save_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "append"):
    """
    Save a DataFrame to a SQLite table.
    
    Parameters:
    - df: the data to save
    - table_name: name of the SQL table
    - if_exists: 'append' adds rows, 'replace' overwrites the table
    """
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"✅ Saved {len(df)} rows to '{table_name}' table.")


def load_from_db(query: str) -> pd.DataFrame:
    """
    Run a SQL query and return results as a DataFrame.
    
    Example: load_from_db("SELECT * FROM news WHERE ticker = 'AAPL'")
    """
    return pd.read_sql(query, engine)


def table_exists(table_name: str) -> bool:
    """Check if a table already exists in the database."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()