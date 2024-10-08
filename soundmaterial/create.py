import argbind

import soundmaterial as sm

def create_db(db_file: str):
    conn = sm.connect(db_file)
    sm.init(conn.cursor())
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_db = argbind.bind(create_db, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        create_db()