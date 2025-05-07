import argbind
import sqlite3
from pathlib import Path
import pandas as pd
from soundmaterial.dataset import train_test_split

import soundmaterial as sm
from tqdm import tqdm

# TODO: we should add a chunk table to the database instead
# and do a one-off processing where, if we know the duration
# we can calculate a table of chunks, each with a chunk_id and 
# audio_file_id, offset (start time), and chunk_duration. 
# then we can just sample from that table instead of recalculating
# the offset each time we load the dataset (which is killer slow)

def split(
        db_file: str, 
        test_size: float = 0.2,
        query: str = None,
        yes: bool = False
    ) -> sqlite3.Connection:
    conn = sm.connect(db_file)

    # confirm that the user wants to do this, as it will clear the existing
    # chunk table if any
    if not yes:
        print(f"WARNING: this will clear the existing split table. Continue? (y/n)")
        if input() != "y":
            return
    

    # get the audio_file w/ it's dataset root
    cursor = conn.cursor()
    conn.execute("begin transaction")

    if query is None:
        query = "SELECT id, path FROM audio_file"

    df = pd.read_sql_query(
        query, conn
    )

    # split the dataset into train and test
    train_df, test_df = train_test_split(df, test_size=test_size)

    df_with_split = train_df.copy()
    df_with_split["split"] = "train"
    test_df["split"] = "test"
    df_with_split = pd.concat([df_with_split, test_df], ignore_index=True)


    # clear the existing chunk table
    conn.execute("DROP TABLE IF EXISTS split")
    sm.create_split_table(conn)

    # insert splits into split table
    for idx, row in tqdm(
        enumerate(df_with_split.itertuples()), 
        total=len(df_with_split), 
        desc="Inserting splits"
    ):
        sm.insert_split(
            conn, 
            sm.Split(row.id, row.split)
        )
    
    # print how many files in each split
    print(f"train: {len(train_df)}")
    print(f"test: {len(test_df)}")

    # commit the changes
    print(f"commit? (y/n)")
    should_commit = input() if not yes else "y"
    if should_commit == "y":
        conn.commit()
        print(f"committed")
    else:
        conn.rollback()


if __name__ == "__main__":
    split = argbind.bind(split, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        split()