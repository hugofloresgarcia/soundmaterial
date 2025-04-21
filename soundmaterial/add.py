import pathlib
from pathlib import Path
import multiprocessing
from typing import Iterable, Sequence, Tuple
import subprocess
import warnings

import argbind
import tqdm
import pandas as pd

import soundmaterial as sm

AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "aiff", "ogg", "mp4", "mov"]


def audio_file_info(path: str) -> dict:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=duration'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    af_format = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=format_name'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    fmt_out, _ = af_format.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        length = float(stdout)
        
        fmt = fmt_out.decode().split('\n')[1].split('=')[-1]
        chan_data = get_audio_channels(path)
        return {
            "path": path,
            "duration": float(length),
            "channels": int(chan_data["channels"]), 
            "format": fmt, 
        }
    except Exception as e:
        warnings.warn(f"Could not process {path}! {e}")
        return None


def get_audio_channels(path: str) -> dict:
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=channels'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode: return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        channels = int(stdout)
        return {"path": path, "channels": int(channels)}
    except:
        return None 


def flatmap(pool: multiprocessing.Pool,
            func: callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def find_audio(path_list: Sequence[str], 
                extensions: Sequence[str] = AUDIO_EXTENSIONS):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
            audios.append(p.rglob(f'*.{ext.upper()}'))
    audios = flatten(audios)
    return audios


def add_dataset(
    db_file: str,
    audio_folder: str, 
    dataset_name: str,
): 
    # we will use multiprocessing to speed up the process
    pool = multiprocessing.Pool(processes=8)

    # connect to the database
    conn = sm.connect(db_file)
    conn.execute("BEGIN TRANSACTION")

    # create the dataset table
    # audio folder needs to be absolute
    audio_folder = Path(audio_folder).resolve()
    if not sm.dataset_exists(conn, dataset_name):
        sm.insert_dataset(conn, sm.Dataset(name=dataset_name, root=audio_folder))
    
    # get the dataset id
    dataset_id = conn.execute(f"SELECT id FROM dataset WHERE name = '{dataset_name}'").fetchone()[0]

    # search for audio files
    audio_files = find_audio([audio_folder])
    audio_lengths = pool.imap_unordered(audio_file_info, audio_files)
    audio_lengths = filter(lambda x: x is not None, audio_lengths)
    audio_lengths = enumerate(audio_lengths)

    # test run
    total_length = 0
    pbar = tqdm.tqdm(audio_lengths,  desc="Processing audio files")
    # pbar = audio_lengths
    for idx, audio_data in pbar:
        af = sm.AudioFile(
            dataset_id=dataset_id,
            path=str(audio_data['path']),
            duration=audio_data['duration'],
            num_channels=audio_data['channels'], 
            format=audio_data['format']
        )
        # insert only if the audio file is not already in the database
        if not sm.audio_file_exists(conn, af.path, dataset_id):
            sm.insert_audio_file(conn.cursor(), af)
        else:
            pbar.set_description(f"audio file {audio_data['path']} already in database")
        total_length += audio_data['duration']
        pbar.set_description(f"dataset length: {pd.to_timedelta(total_length, unit='s')}")
    pbar.close()

    conn.commit()
    conn.close()


if __name__ == "__main__":
    import argbind

    add_dataset = argbind.bind(add_dataset, without_prefix=True, positional=True)

    args = argbind.parse_args()

    with argbind.scope(args):
        add_dataset()
