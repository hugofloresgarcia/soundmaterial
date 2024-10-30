import subprocess
from pathlib import Path
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
import multiprocessing as mp

def convert_one(input_path, output_path, audio_file):
    # Define output file path and maintain directory structure
    relative_path = audio_file.relative_to(input_path)
    output_file = output_path / relative_path.with_suffix('.wav')
    
    # Create output directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        print(f"Skipping {audio_file} as {output_file} already exists.")
        return 

    command = [
        'ffmpeg', '-i', str(audio_file), '-y', str(output_file)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Converted {audio_file} to {output_file}")

def convert_to_wav(input_dir, output_dir, debug=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all audio files with extensions (mp3, wav, ogg, aac)
    from soundmaterial.add import find_audio
    print(f"looking for audio files")
    audio_files = list(find_audio([input_path]))
    print(f"Found {len(audio_files)} audio files.")

    # Convert audio files to WAV format
    pool = mp.Pool(mp.cpu_count())
    if not debug:
        pool.map(partial(convert_one, input_path, output_path), audio_files)
    else:
        for audio_file in tqdm(audio_files):
            convert_one(input_path, output_path, audio_file)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert audio files to FLAC format.")
    parser.add_argument("input_dir", help="The input directory containing audio files.")
    parser.add_argument("output_dir", help="The output directory where FLAC files will be saved.")

    # Parse command line arguments
    args = parser.parse_args()

    # Start the conversion
    convert_to_wav(args.input_dir, args.output_dir)

