import argbind
import sqlite3
from pathlib import Path
import pandas as pd

import soundmaterial as sm

def summarize(conn, query: str = None, chunks: bool = False) -> sqlite3.Connection:
    if query is None:
        if chunks:
            query = "SELECT * FROM chunk JOIN audio_file as af ON chunk.audio_file_id = af.id JOIN dataset ON af.dataset_id = dataset.id"
        else:
            query = "SELECT * FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id"

    df = pd.read_sql_query(query, conn)

    print(f"Loaded {len(df)} rows from {conn}")
    # show audio_file counts per dataset
    print(df["name"].value_counts())
    return df

def summarize_db(db_file: str, query: str = None, chunks: bool = False) -> sqlite3.Connection:
    if query is None:
        if chunks:
            query = "SELECT * FROM chunk JOIN audio_file as af ON chunk.audio_file_id = af.id JOIN dataset ON af.dataset_id = dataset.id"
        else:
            query = "SELECT * FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id"

    conn = sm.connect(db_file)
    summarize(conn, query, chunks)
    breakpoint()

    
if __name__ == "__main__":
    summarize = argbind.bind(summarize, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        summarize()