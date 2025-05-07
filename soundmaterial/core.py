import sqlite3
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# ~~~~ connect to the database ~~~~

def connect(db_file: str, create=False) -> sqlite3.Connection:
    """
    connect to the database.
    """
    if not Path(db_file).exists() and not create:
        raise FileNotFoundError(f"Database file {db_file} does not exist.")
    # print(f"connecting to database at {db_file}")
    conn = sqlite3.connect(db_file)

    # print all the tables in the database
    # print(f"loaded database from {db_file}")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print("tables in the database:")
    # print(cur.fetchall())

    return conn


# ~~~~ execute a query and get the result as a dataframe
def df_query(conn, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)

# ~~~~ df: get the audio file table w/ the dataset table ~~~~
def df_audio_file(conn) -> pd.DataFrame:
    return df_query(conn, "SELECT * FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id")

# ~~~~ df: get the chunk table w/ the audio_file and dataset table ~~~~
def df_chunk(conn) -> pd.DataFrame:
    return df_query(conn, "SELECT * FROM chunk as chunk JOIN audio_file as af ON chunk.audio_file_id = af.id JOIN dataset ON af.dataset_id = dataset.id")

# ~~~~~~~~~~~~~~~~
# ~~~~ tables ~~~~
# ~~~~~~~~~~~~~~~~

# ~~~~ dataset ~~~~

@dataclass
class Dataset:
    name: str
    root: str
def create_dataset_table(cur):
    cur.execute(
        """
        CREATE TABLE dataset (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name STRING NOT NULL UNIQUE,
            root STRING,
            UNIQUE (name, root)
        )
        """
    )

def insert_dataset(cur, dataset: Dataset):
    return cur.execute(
        f"""
        INSERT INTO dataset (
            name, root
        ) VALUES (
            '{dataset.name}', '{dataset.root}'
        )
        RETURNING id
        """
    ).fetchone()[0]

def get_dataset(cur, name: str) -> Dataset:
    return cur.execute(f"""
        SELECT id, root
        FROM dataset
        WHERE name = '{name}'
    """).fetchone()

def dataset_exists(cur, name: str) -> bool:
    return cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM dataset
            WHERE name = '{name}'
        )
    """).fetchone()[0]

# ~~~~ audio file ~~~~
@dataclass
class AudioFile:
    dataset_id: int
    path: Optional[str] = None
    duration: Optional[float] = None
    num_channels: Optional[int] = None
    format: Optional[str] = None

def create_audio_file_table(cur):
    cur.execute(
        """
        CREATE TABLE audio_file (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            path STRING,
            dataset_id INTEGER NOT NULL,
            duration REAL,
            num_channels INTEGER,
            format STRING,
            FOREIGN KEY (dataset_id) REFERENCES dataset(id),
            UNIQUE (path, dataset_id)
        )
        """
    )

def insert_audio_file(cur, audio_file: AudioFile):
    return cur.execute(
        """
        INSERT INTO audio_file (
            path, dataset_id, duration, num_channels, format
        ) VALUES (
            ?, ?, ?, ?, ?
        )
        RETURNING id
        """,
        (audio_file.path, audio_file.dataset_id, audio_file.duration, 
        audio_file.num_channels, audio_file.format)
    ).fetchone()[0]



# def create_audio_file_table(cur):
#     cur.execute(
#         """
#         CREATE TABLE audio_file (
#             id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
#             path STRING,
#             dataset_id INTEGER NOT NULL,
#             num_frames INTEGER,
#             sample_rate INTEGER,
#             num_channels INTEGER,
#             encoding STRING,
#             FOREIGN KEY (dataset_id) REFERENCES dataset(id),
#             UNIQUE (path, dataset_id)
#         )
#         """
#     )


# def insert_audio_file(cur, audio_file: AudioFile):
#     return cur.execute(
#         """
#         INSERT INTO audio_file (
#             path, dataset_id, num_frames, sample_rate, num_channels, encoding
#         ) VALUES (
#             ?, ?, ?, ?, ?, ?, ?
#         )
#         RETURNING id
#         """,
#         (audio_file.path, audio_file.dataset_id, audio_file.num_frames, 
#         audio_file.sample_rate, audio_file.num_channels, 
#         audio_file.encoding)
#     ).fetchone()[0]

#     # convert to a dataframe and insert with dataframe
#     df = pd.DataFrame([audio_file.__dict__])
#     cur.insert("audio_file", df)


def get_audio_file_table(cur, dataset_id: int) -> pd.DataFrame:
    return cur.execute(f"""
        SELECT *
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).df()

def audio_file_exists(cur, path: str, dataset_id: int) -> bool:
    # return cur.execute(f"""
    #     SELECT EXISTS(
    #         SELECT 1
    #         FROM audio_file
    #         WHERE path = '{path}'
    #         AND dataset_id = {dataset_id}
    #     )
    # """).fetchone()[0]
    return cur.execute("""
        SELECT EXISTS(
            SELECT 1
            FROM audio_file
            WHERE path = ?
            AND dataset_id = ?
        )
    """, (path, dataset_id)).fetchone()[0]



# ~~~~ control signal ~~~~
@dataclass
class ControlSignal:
    path: str
    audio_file_id: int
    name: str
    sample_rate: int
    hop_size: int
    num_frames: int
    num_channels: int
def create_ctrl_sig_table(cur):
    cur.execute(
        """
        CREATE TABLE ctrl_sig (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            path STRING NOT NULL UNIQUE,
            audio_file_id INTEGER NOT NULL,
            name STRING NOT NULL,
            sample_rate INTEGER NOT NULL,
            hop_size INTEGER NOT NULL,
            num_frames INTEGER NOT NULL,
            num_channels INTEGER NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id)
        )
        """
    )

def insert_ctrl_sig(cur, ctrl_sig: ControlSignal):
    return cur.execute(
        """
        INSERT INTO ctrl_sig (
            path, audio_file_id, name, sample_rate, hop_size, num_frames, num_channels
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?
        )
        RETURNING id
        """,
        (ctrl_sig.path, ctrl_sig.audio_file_id, ctrl_sig.name, ctrl_sig.sample_rate,
        ctrl_sig.hop_size, ctrl_sig.num_frames, ctrl_sig.num_channels)
    ).fetchone()[0]

def ctrl_sig_exists(cur, path: str, audio_file_id: int) -> bool:
    # print(f'looking for ctrl_sig {path} for audio_file_id {audio_file_id}')
    out =  cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM ctrl_sig
            WHERE path = '{path}'
            AND audio_file_id = {audio_file_id}
        )
    """).fetchone()[0]
    # print(f'ctrl_sig exists: {out}')
    return out

# a table for train/test splits
@dataclass
class Split:
    audio_file_id: int
    split: str

def create_split_table(cur):
    cur.execute(
        """
        CREATE TABLE split (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            audio_file_id INTEGER NOT NULL,
            split STRING NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, split)
        )
        """
    )

def insert_split(cur, split: Split):
    # breakpoint()
    cur.execute(
        f"""
        INSERT INTO split ( audio_file_id, split ) 
        VALUES ({split.audio_file_id}, '{split.split}')
        """
    )


@dataclass
class Caption:
    text: str
    audio_file_id: int

def create_caption_table(cur):
    cur.execute(
        """
        CREATE TABLE caption (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            text STRING NOT NULL,
            audio_file_id INTEGER NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id)
        )
        """
    )

def insert_caption(cur, caption: Caption):
    cur.execute(
        """
        INSERT INTO caption ( text, audio_file_id ) 
        VALUES (?, ?)
        """,
        (caption.text, caption.audio_file_id)
    )


@dataclass
class Chunk:
    audio_file_id: int
    offset: float
    duration: float

def create_chunk_table(cur):
    cur.execute(
        """
        CREATE TABLE chunk (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            audio_file_id INTEGER NOT NULL,
            offset REAL NOT NULL,
            duration REAL NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, offset)
        )
        """
    )

def insert_chunk(cur, chunk: Chunk):
    cur.execute(
        f"""
        INSERT INTO chunk ( audio_file_id, offset, duration ) 
        VALUES ({chunk.audio_file_id}, {chunk.offset}, {chunk.duration})
        """
    )

def chunk_exists(cur, audio_file_id: int, offset: float) -> bool:
    return cur.execute(f"""
        SELECT EXISTS(
            SELECT 1
            FROM chunk
            WHERE audio_file_id = {audio_file_id}
            AND offset = {offset}
        )
    """).fetchone()[0]

@dataclass
class Split:
    audio_file_id: int
    split: str

def create_split_table(cur):
    cur.execute(
        """
        CREATE TABLE split (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            audio_file_id INTEGER NOT NULL,
            split STRING NOT NULL,
            FOREIGN KEY (audio_file_id) REFERENCES audio_file(id),
            UNIQUE (audio_file_id, split)
        )
        """
    )

def insert_split(cur, split: Split):
    cur.execute(
        f"""
        INSERT INTO split ( audio_file_id, split ) 
        VALUES ({split.audio_file_id}, '{split.split}')
        """
    )

def get_split(cur, audio_file_id: int) -> pd.DataFrame:
    return cur.execute(f"""
        SELECT *
        FROM split
        WHERE audio_file_id = {audio_file_id}
    """).fetchall()



def init(cursor: sqlite3.Cursor):
    for fn in [
        create_dataset_table,
        create_audio_file_table,
        create_ctrl_sig_table,
        create_split_table, 
        create_caption_table, 
        create_chunk_table
    ]:
        cur = cursor
        try: 
            print(f"running {fn.__name__}")
            fn(cur)
        except Exception as e:
            print(f"error: {e}")

    print("done! :)")


def test_db():
    # create a databse in memory, grab a cursor
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # init the database
    init(cur)

    # insert a dataset
    dataset = Dataset(name="test", root="test")
    dataset_id = insert_dataset(cur, dataset)

    # insert an audio file
    audio_file = AudioFile(dataset_id=dataset_id, path="test", num_frames=1, sample_rate=4800, num_channels=2, bit_depth=3, encoding="test")
    audio_file_id = insert_audio_file(cur, audio_file)

    # insert a control signal
    ctrl_sig = ControlSignal(path="rms", audio_file_id=audio_file_id, name="test", sample_rate=20, hop_size=-1, num_frames=1, num_channels=1)
    ctrl_sig_id = insert_ctrl_sig(cur, ctrl_sig)

    # insert a split
    split = Split(audio_file_id=audio_file_id, split="train")
    insert_split(cur, split)
    split = Split(audio_file_id=audio_file_id, split="validation")
    insert_split(cur, split)

    # insert a caption
    caption = Caption(text="test", audio_file_id=audio_file_id)
    insert_caption(cur, caption)

    # print the tables
    print("dataset table:")
    print(pd.read_sql_query("SELECT * FROM dataset", conn))
    print("audio_file table:")
    print(pd.read_sql_query("SELECT * FROM audio_file", conn))
    print("ctrl_sig table:")
    print(pd.read_sql_query("SELECT * FROM ctrl_sig", conn))
    print("split table:")
    print(pd.read_sql_query("SELECT * FROM split", conn))
    print("caption table:")
    print(pd.read_sql_query("SELECT * FROM caption", conn))
    # drop the tables
    cur.execute("DROP TABLE dataset")
    cur.execute("DROP TABLE audio_file")
    cur.execute("DROP TABLE ctrl_sig")
    cur.execute("DROP TABLE split")
    cur.execute("DROP TABLE caption")
    # close the connection
    conn.close()
    print("done! :)")

if __name__ == "__main__":
    test_db()