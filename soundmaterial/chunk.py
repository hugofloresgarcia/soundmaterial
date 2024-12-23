import argbind
import sqlite3
from pathlib import Path

import soundmaterial as sm
from tqdm import tqdm

# TODO: we should add a chunk table to the database instead
# and do a one-off processing where, if we know the duration
# we can calculate a table of chunks, each with a chunk_id and 
# audio_file_id, offset (start time), and chunk_duration. 
# then we can just sample from that table instead of recalculating
# the offset each time we load the dataset (which is killer slow)

def chunk(
        db_file: str, 
        chunk_duration: float
    ) -> sqlite3.Connection:
    conn = sm.connect(db_file)

    # confirm that the user wants to do this, as it will clear the existing
    # chunk table if any
    print(f"WARNING: this will clear the existing chunk table. Continue? (y/n)")
    if input() != "y":
        return
    
    # clear the existing chunk table
    conn.execute("DROP TABLE IF EXISTS chunk")
    sm.create_chunk_table(conn)

    # get the audio_file w/ it's dataset root
    cursor = conn.cursor()
    conn.execute("begin transaction")
    cursor.execute("SELECT audio_file.id, audio_file.duration, audio_file.path, dataset.root FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id")

    for row in tqdm(cursor.fetchall()):
        id, duration, path, root = row
        num_chunks = int(duration / chunk_duration)
        print(f"creating {num_chunks} chunks for {path}")
        for i in range(num_chunks):
            offset = i * chunk_duration
            assert offset + chunk_duration <= duration
            chunk = sm.Chunk(id, offset, chunk_duration)
            chunk = sm.insert_chunk(conn, chunk)

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    print(f" created {total_chunks} chunks")
    print(f"commit? (y/n)")
    if input() == "y":
        conn.commit()
        print(f"committed")
    else:
        conn.rollback()
    
if __name__ == "__main__":
    chunk = argbind.bind(chunk, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        chunk()