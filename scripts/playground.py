import os
from pathlib import Path
import shutil

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import soundmaterial as sm

# the path to our database
db_path = "/home/hugo/soup/sm.db"

# connect to our database
conn = sm.connect(db_path)

# find all the wav files
query = """
    SELECT * FROM audio_file
    JOIN dataset ON audio_file.dataset_id = dataset.id
    WHERE format = 'wav'
"""

# Create a subset of the database
subset = pd.read_sql_query(query, conn)
print(f"Creating a subset of {len(subset)} rows")
print(subset)

# create a row called "folder", which is the parent folder of the audio file
subset['folder'] = subset['path'].apply(lambda x: os.path.dirname(x))

print("do stuff here!")
breakpoint()
