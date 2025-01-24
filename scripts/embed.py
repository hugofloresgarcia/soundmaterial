import msclap
import pandas as pd
from typing import List

import numpy as np
import tqdm

import soundmaterial as sm
from soundmaterial.utils import dim_reduce

from msclap import CLAP
# arguments
DB_PATH = "clack.db"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clap_model = CLAP(version = '2023', use_cuda=True)

# connect to the sm database
conn = sm.connect(DB_PATH)

# how many examples to use for the embedding
subset_size = 5000 

# load the audio and caption tables, join on audio file id, 
# also add a dataset_id and dataset_name
tbl = pd.read_sql_query(
    """
    SELECT 
        audio_file.id as id,
        audio_file.path as path,
        caption.text as text,
        dataset.id as dataset_id,
        dataset.name as dataset_name
    FROM audio_file
    JOIN caption ON audio_file.id = caption.audio_file_id
    JOIN dataset ON audio_file.dataset_id = dataset.id
    """,
    conn
) 

# grab a random subset of the table
tbl = tbl.sample(subset_size)

# grab the captions
captions = tbl["text"].tolist()

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
    n_components=2
)

print(f"done! :)")