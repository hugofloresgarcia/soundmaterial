import pytest

import soundmaterial as sm
from soundmaterial.dsp.signal import Signal

def test_soundmaterial():
    from soundmaterial.create import create_db
    conn = create_db("test.db")

    assert conn is not None
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    from soundmaterial.add import add_dataset
    add_dataset("test.db", "./assets/guitar-abstract", "guitar-abstract")

    # make sure our audio_file table is not empty
    cursor.execute("SELECT * FROM audio_file")
    audio_files = cursor.fetchall()

    assert len(audio_files) > 0

    # make sure our dataset table is not empty
    cursor.execute("SELECT * FROM dataset")
    datasets = cursor.fetchall()
    assert len(datasets) > 0

    # make sure our dataset table has the correct name
    cursor.execute("SELECT name FROM dataset WHERE id = 1")
    dataset_name = cursor.fetchone()[0]
    assert dataset_name == "guitar-abstract"

    # try cutting the dataset into 5 second chunks
    from soundmaterial.chunk import chunk
    chunk("test.db", 5.0, yes=True)
    
    # get dataframe from the db
    import pandas as pd
    df = pd.read_sql_query("SELECT * FROM audio_file", conn)

    # try creating a dataset from the db
    from soundmaterial.dataset import Dataset

    dataset = Dataset(df)

    # get some dataset items
    item = dataset[0]
    assert item is not None
    assert isinstance(item["sig"], Signal)
    assert isinstance(item["sig_dry"], Signal)
    assert isinstance(item["sig"].metadata, dict)
    assert "path" in item["sig"].metadata 

    # make splits
    from soundmaterial.train_test_split import split
    split("test.db", test_size=0.2, yes=True)

    # get the splits
    cursor.execute("SELECT * FROM split")
    splits = cursor.fetchall()
    assert len(splits) > 0
