import os
from pathlib import Path
import shutil

import pandas as pd
import soundmaterial as sm

def create_subset(
    db_path: str,
    output_folder: str,
    query: str,
    symlinks: bool = False
) -> str:
    """ create a subset of a dataset by copying or symlinking the audio files to a new folder. 
    The new subset is saved to a new database file.

    The query is a SQL query that selects the rows to keep in the subset.
    
    Parameters:
    db_path (str): path to the database file
    output_folder (str): path to the output folder
    query (str): a SQL query that selects the rows to keep in the subset

    Returns:
    str: path to the new database file
    
    """
    conn = sm.connect(db_path)

    # Create a subset of the database
    subset = pd.read_sql_query(query, conn)
    print(f"Creating a subset of {len(subset)} rows")
    print(subset)

    # print the total duration
    total_duration = subset['duration'].sum()
    print(f"Total duration: {total_duration} seconds")

    print(f"hours: {total_duration / 3600}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # copy all the audio files to the output folder
    for i, row in subset.iterrows():
        audio_file = row['path']
        outfile = os.path.join(output_folder, os.path.basename(audio_file))
        if os.path.exists(outfile):
            continue

        if symlinks:
            os.symlink(audio_file, outfile)
        else:
            shutil.copy(audio_file, outfile)

    # save a db file with the subset to the output folder
    subset_db = os.path.join(output_folder, "subset.db")
    subset_conn = sm.connect(subset_db)

    subset.to_sql("audio_file", subset_conn, index=False)

    # test a connection
    test_conn = sm.connect(subset_db)
    # test audio files
    audio_files = pd.read_sql_query("SELECT * FROM audio_file", test_conn)
    print(audio_files)

    return subset_db

        

if __name__ == "__main__":

    # search for all audio files with the word 'animal' in the description
    # query = "SELECT * FROM audio_file WHERE path LIKE '%insect%'"
    # query = query + " OR path LIKE '%bird%'"
    # query = query + " OR path LIKE '%growl%'"
    query = "SELECT * FROM audio_file WHERE path LIKE '%Pianos - Prepared%'"

    create_subset(
        db_path="/home/hugo/soundmaterial/sm.db",
        output_folder="/home/hugo/rave/data/subset-prep-piano",
        query=query,
        symlinks=True
    )

    