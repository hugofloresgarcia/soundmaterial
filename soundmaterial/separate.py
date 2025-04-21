import sqlite3
from pathlib import Path
import subprocess

import argbind
from tqdm import tqdm
import soundmaterial as sm
from soundmaterial.add import add_dataset


def separate(
        db_file: str, 
        name: str, # name of the dataset to separate
        # output_folder: str, 
    ) -> sqlite3.Connection:

    # check if there's a "separated" folder in the current directory
    # if its there, exit and warn the user to remove it because it will be removed

    if Path("separated").exists():
        print("separated folder already exists! Please remove it before continuing.")
        return

    conn = sm.connect(db_file)

    new_dataset_name = name + "-separated"

    # get the audio_file w/ it's dataset root
    cursor = conn.cursor()
    conn.execute("begin transaction")
    cursor.execute("SELECT audio_file.id, audio_file.duration, audio_file.path, dataset.root FROM audio_file JOIN dataset ON audio_file.dataset_id = dataset.id WHERE dataset.name = ?", (name,))

    for row in tqdm(cursor.fetchall()):
        id, duration, path, root = row
        audio_file_path = Path(root) / path
        try:
            subprocess.check_call(["demucs", str(audio_file_path)])    
        except subprocess.CalledProcessError as e:
            print(f"Error separating {audio_file_path}: {e}")
            continue
    
    # check that there's a folder called "separated" in the current dir
    if not Path("separated").exists():
        print("separated folder not found! Please check if demucs was run correctly.")
        return

    print(f"all done! your files are in the 'separated' folder. you can now add them to the database.")



    
if __name__ == "__main__":
    separate = argbind.bind(separate, without_prefix=True, positional=True)

    args = argbind.parse_args()
    with argbind.scope(args):
        separate()