import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp

def remove_duplicates(input_dir):
    input_path = Path(input_dir)

    # Find all audio files with extensions (mp3, wav, ogg, aac)
    from soundmaterial.add import find_audio
    audio_files = find_audio([input_path])

    # Convert audio files to WAV format
    pool = mp.Pool(mp.cpu_count())
    for audio_file in tqdm(audio_files):
        # check if the file is a duplicate of a wav file
        if audio_file.suffix != '.wav':
            if audio_file.with_suffix('.wav').exists():
                print(f"Removing {audio_file} as {audio_file.with_suffix('.wav')} already exists.")
                audio_file.unlink()
                continue
        


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert audio files to FLAC format.")
    parser.add_argument("input_dir", help="The input directory containing audio files.")

    # Parse command line arguments
    args = parser.parse_args()

    # Start the conversion
    remove_duplicates(args.input_dir)

