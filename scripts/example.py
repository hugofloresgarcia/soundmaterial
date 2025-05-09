import os
from pathlib import Path
import shutil

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import soundmaterial as sm

# the path to our database
db_path = "sm.db"

# create a new database
conn = sm.create.create_db(db_path)

# add a dataset to the database
sm.add.add_dataset(db_path, "./assets/guitar-abstract", "guitar-abstract")

# find all the wav files
query = "SELECT * FROM audio_file WHERE format = 'wav'"

# Create a subset of the database
subset = pd.read_sql_query(query, conn)
print(f"Creating a subset of {len(subset)} rows")
print(subset)

# create a row called "folder", which is the parent folder of the audio file
subset['folder'] = subset['path'].apply(lambda x: os.path.dirname(x))

# print the total duration
total_duration = subset['duration'].sum()
print(f"Total duration: {total_duration} seconds")
print(f"hours: {total_duration / 3600}")

# what are the quartiles of the duration?
print(subset['duration'].describe())

# how many over 5 minutes? 
print(f"Over 5 minutes: {len(subset[subset['duration'] > 300])}")

# how many under 5 seconds?
print(f"Under 5 seconds: {len(subset[subset['duration'] < 5])}")

# how many under 1 second? 
print(f'Under 1 second: {len(subset[subset["duration"] < 1])}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# vizualization
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# what's the relationship between duration and folder? 

sns.boxplot(x='folder', y='duration', data=subset)
plt.savefig("duration_vs_folder.png")

# make a histogram of durations
sns.histplot(subset['duration'])
plt.savefig("duration_histogram.png")
