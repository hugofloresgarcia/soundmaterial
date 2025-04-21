import random 
import sqlite3

import numpy as np
import gradio as gr
import pandas as pd
import soundmaterial.dsp.signal as sn

import soundmaterial as sm

import argparse


# open the db 
def to_output(sig: sn.Signal):
    wave = sn.to_mono(sig).wav[0][0].numpy()
    return sig.sr, wave * np.iinfo(np.int16).max
    

def search_audio_by_filename(query):
    conn = sm.connect(db_path)
    cur = conn.cursor()

    cur.execute(f"SELECT id, path FROM audio_file WHERE path LIKE ?", (f"%{query}%",))
    results = cur.fetchall()

    if not results:
        raise gr.Error(f"no results found")

    # return a random result
    audio_id, path = random.choice(results)
    sig = sn.excerpt(path, duration=10.0)

    # get a dataframe with the rest of the columns
    df = pd.read_sql_query(
        f"SELECT * FROM audio_file WHERE id={audio_id}",
        conn
    )

    return path, to_output(sig), df, f"{len(results)} results found"

with gr.Blocks() as demo:
    
    search_box = gr.Textbox(label="Search for a file in the dataset. leave blank for random.")
    search_button = gr.Button("Search")
    
    filename = gr.Label(label="Filename")
    audio_result = gr.Audio(label="Audio", type="numpy")
    df_result = gr.DataFrame()

    num_results_found = gr.Label(label="")

    # search button functionality
    search_button.click(
        fn=search_audio_by_filename,
        inputs=search_box,
        outputs=[filename, audio_result, df_result, num_results_found]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_file" , default="sm.db")
    args = parser.parse_args()
    db_path = args.db_file

    demo.queue()
    demo.launch()
