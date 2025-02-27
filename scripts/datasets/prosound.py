from pathlib import Path

import pandas as pd

import soundmaterial as sm
from soundmaterial.create import create_db
from soundmaterial.add import add_dataset
from soundmaterial.subset import create_subset

path_to_prosound = Path("/media/pancho/prosound_core_complete/")
db_path = "/media/pancho/sm.db"

# create a database
create_db(db_path)

# what are the subpaths under prosound? 
subfolders = [
    p for p in path_to_prosound.iterdir() if p.is_dir()
]
print(f"found {len(subfolders)} subfolders")

# add the subfolders to the database
for i, subfolder in enumerate(subfolders):
    print(f"adding {subfolder} to the database as {Path(subfolder).stem}")
    try:
        # check if we already have this dataset in the database
        # if we do, skip it
        dataset_tbl = pd.read_sql_query("SELECT * FROM dataset", sm.connect(db_path))
        if Path(subfolder).stem in dataset_tbl["name"].values:
            print(f"skipping {subfolder}")
            continue
        add_dataset(
            db_path, 
            str(subfolder), 
            Path(subfolder).stem
        )
    except Exception as e:
        print(f"error: {e}")

# TODO: process captions 
# here, we would add the captions to the caption table inthe database
from soundmaterial.core import Caption, insert_caption
conn = sm.connect(db_path)

def filename2caption(path: str):
    caption = Path(path).stem
    # remove underscores
    caption = caption.replace("_", " ")
    return caption

# go through all the filenames, and insert a caption
# based on the filename
audio_file_tbl = pd.read_sql_query("SELECT * FROM audio_file", conn)

cur = conn.cursor()
cur.execute("BEGIN TRANSACTION")

for i, row in audio_file_tbl.iterrows():
    caption = filename2caption(row["path"])
    caption = Caption(text=caption, audio_file_id=row["id"])
    insert_caption(cur, caption)

conn.execute("COMMIT")

# print a join of the audio_file and caption tables
query = """
    SELECT * FROM audio_file
    JOIN caption
    ON audio_file.id = caption.audio_file_id
"""
print(pd.read_sql_query(query, conn))


# TODO (layton):
# here, instead of making a single subset, we would make multiple subsets
# for different classes of sounds (e.g. speech, machines, animals, musical instruments, synthesizers, etc.) for data balancing

# # now, let's create a subset for training
# # for our subset, let's only keep the files 
# # whose duration is shorter than 5 minutes (for dataloading speed purposes)
# query = "SELECT * FROM audio_file WHERE duration < 300 and format = 'wav'"
# output_folder = "data/prosound_training_subset_symlinks"
# create_subset(
#     db_path,
#     output_folder,
#     query,
#     symlinks=True # don't copy the files, just symlink!
# )
