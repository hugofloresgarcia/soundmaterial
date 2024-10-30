import argbind
import sqlite3

import soundmaterial as sm

def create_db(db_file: str) -> sqlite3.Connection:
    conn = sm.connect(db_file)
    sm.init(conn.cursor())
    conn.commit()

    return conn

if __name__ == "__main__":
    create_db = argbind.bind(create_db, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        create_db()