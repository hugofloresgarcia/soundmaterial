"""preliminary captioning script. for now, uses the filename as the caption."""
from pathlib import Path

import sqlite3


import soundmaterial as sm
from tqdm import tqdm


def caption(db_path: str, filename: bool = True, yes: bool = False) -> sqlite3.Connection:

    assert filename
    conn = sm.connect(db_path)

    # get the audio_file w/ it's dataset root
    cursor = conn.cursor()
    conn.execute("begin transaction")
    cursor.execute(
        "SELECT audio_file.id, audio_file.path, dataset.root FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id"
    )

    rows = cursor.fetchall()
    print(f"found {len(rows)} audio files")

    if not yes:
        print(f"WARNING: this will clear the existing caption table. Continue? (y/n)")
        if input() != "y":
            return
    # create a caption table
    conn.execute("DROP TABLE IF EXISTS caption")
    sm.create_caption_table(conn)

    print("created caption table")
    # for each row, create a caption
    for row in tqdm(rows):
        id, path, root = row
        # get the filename
        filename = Path(path).stem
        # create a caption
        caption = sm.Caption(filename, id)
        # insert the caption into the database
        caption = sm.insert_caption(conn, caption)

    # commit the transaction
    if not yes:
        print(f"WARNING: this will clear the existing caption table. Continue? (y/n)")
        if input() != "y":
            conn.rollback()
            return
    else:
        print(f"WARNING: this will clear the existing caption table. Continuing...")

    conn.commit()
    print("committed transaction")
    # close the connection
    conn.close()
    return conn





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Caption the audio files in the database")
    parser.add_argument(
        "db_path", type=str, default="sm.db", help="Path to the database file"
    )
    parser.add_argument(
        "--filename", action="store_true", help="Use the filename as the caption"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation for dropping tables"
    )

    args = parser.parse_args()

    caption(args.db_path, args.filename, args.yes)