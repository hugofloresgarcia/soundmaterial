# soundmaterial
a python library for examining (with your eyes and ears) large sound datasets with a playful curiosity 

## setup
first, install audiotools
```bash
git clone  https://github.com/hugofloresgarcia/audiotools.git
cd audiotools
pip install -e .
```

now, we can install soundmaterial
```bash
git clone  https://github.com/hugofloresgarcia/soundmaterial
cd soundmaterial
pip install -e .
```

## usage 

see `scripts/example.py` for an example. 

## cli

create a new database called `sm.db`
```bash
python -m soundmaterial.create ./sm.db
```

add a folder of sounds to the database
```bash
python -m soundmaterial.add ./sm.db /path/to/sounds
```

open a web interface and search for sounds by filename
```bash
python -m soundmaterial.listen ./sm.db
```

create a subset (copy) of the audio files with an SQL query (use `--symlinks` to create symlinks instead of copying files)
```bash
python -m soundmaterial.subset ./sm.db "SELECT * FROM audio_file WHERE duration < 300" --output_folder data/subset --symlinks
```

examine and edit the dataset tables in a web browser
```
pip install sqlite-web
sqlite_web ./sm.db
```


## the prosound dataset
there is a script for processing the audio and captions in the IAL's prosound dataset. 
you can run it with the following command.

```bash
python scripts/datasets/prosound.py
```

you may have to modify the paths in the script to point to the correct locations of the prosound dataset on your machine.

