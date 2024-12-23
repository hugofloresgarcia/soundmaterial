import argbind
import sqlite3
from pathlib import Path

import soundmaterial as sm

def clean_db(db_file: str) -> sqlite3.Connection:
    conn = sm.connect(db_file)

    # get the audio_file w/ it's dataset root
    cursor = conn.cursor()
    cursor.execute("SELECT audio_file.id, audio_file.path, dataset.root FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id")
    for row in cursor.fetchall():
        id, path, root = row
        audio_file_path = Path(root) / path
        if not audio_file_path.exists():
            print(f"Removing {audio_file_path} from database")
            cursor.execute("DELETE FROM audio_file WHERE id = ?", (id,))

    print(f"commit? (y/n)")
    if input() == "y":
        conn.commit()
    else:
        conn.rollback()
    
if __name__ == "__main__":
    clean_db = argbind.bind(clean_db, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        clean_db()