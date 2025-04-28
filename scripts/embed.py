import msclap
import pandas as pd
from typing import List

import numpy as np
import tqdm

import soundmaterial as sm
from soundmaterial.utils import dim_reduce

from msclap import CLAP
# arguments
DB_PATH = "sm.db"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clap_model = CLAP(version = '2023', use_cuda=True)

# connect to the sm database
conn = sm.connect(DB_PATH)

# how many examples to use for the embedding (max)
subset_size = 5000 

# load the audio and caption tables, join on audio file id, 
# also add a dataset_id and dataset_name
tbl = pd.read_sql_query(
    """
    SELECT af.id, af.path, af.duration, af.dataset_id, ds.name as dataset_name, 
           ac.text
    FROM audio_file as af
    JOIN caption as ac ON ac.audio_file_id = af.id
    JOIN dataset as ds ON ds.id = af.dataset_id
    """, conn)

# grab a random subset of the table
tbl = tbl.sample(min(len(tbl), subset_size))

# grab the captions
captions = tbl["text"].tolist()

# if less than 100 captions, repeat the captions
if len(captions) < 100:
    captions = captions * (100 // len(captions) + 1)
    captions = captions[:100]

# Extract text embeddings
embeddings = []
for cap in tqdm.tqdm(captions):
    emb = clap_model.get_text_embeddings([cap]).cpu().numpy()
    embeddings.append(emb)
embeddings = np.concatenate(embeddings, axis=0)
print(f"got embeddings with shape {embeddings.shape}")


# do a dim reduction and save the plot to html
dim_reduce(
    embeddings, 
    ["unknown" for _ in range(embeddings.shape[0])],
    save_path="test.html",
    method="tsne",
    title="clap embeddings", 
    metadata=[{"cap": cap} for cap in captions],
    n_components=2, 
)

print(f"done! :)")