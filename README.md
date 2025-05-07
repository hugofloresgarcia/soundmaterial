# soundmaterial
a python library for examining (with your eyes and ears) large sound datasets with a playful curiosity 

## setup
first, install vampnet (TODO: need to fix this circular dependency by factoring out vampnet's signal utils)
```bash
git clone -b gloop  https://github.com/hugofloresgarcia/vampnet.git 
cd vampnet
pip install -e .
```

now, we can install soundmaterial
```bash
git clone  https://github.com/hugofloresgarcia/soundmaterial
cd soundmaterial
pip install -e .
```

## usage 

### usage example
see `scripts/example.py` for an example. 

### test stuff
```bash
pip install pytest
python -m pytest
```

### visualize CLAP embeddings
visualize CLAP embeddings on a dataset. 
see `scripts/embed.py`

**note**: you'll need to install some extra dependencies: 
```
pip install msclap --no-deps
pip install plotly transformers umap-learn scikit-learn torchlibrosa 
```

## cli

### new database
create a new database called `sm.db`
```bash
python -m soundmaterial.create ./sm.db
```

### add audio files
add a folder of sounds to the database
```bash
python -m soundmaterial.add ./sm.db /path/to/sounds
```

### listen to samples from the dataset
open a web interface and search for sounds by filename
```bash
python -m soundmaterial.listen ./sm.db
```

chunk the audio files into 5 second chunks (in a new table called `chunk`)
```bash
python -m soundmaterial.chunk ./sm.db 5.0 # duration in seconds
```

create a train test split, split by audio files
```bash
python -m soundmaterial.split ./sm.db  --test_size 0.2 --query "SELECT * FROM audio_file" 
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
