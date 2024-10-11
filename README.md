# soundmaterial
a python library for examining (with your eyes and ears) large sound datasets with a playful curiosity 

## setup
```bash
git clone  https://github.com/hugofloresgarcia/soundmaterial
cd soundmaterial
git submodule update --init --recursive
pip install -e .
```

install audiotools
```bash
pip install -e lib/audiotools
```

## doing things

create a new database
```bash
python -m soundmaterial.create ./sm.db
```

add a folder of sounds to the database
```bash
python -m soundmaterial.add ./sm.db /path/to/sounds
```

listen to sounds and search by filename
```bash
python -m soundmaterial.listen ./sm.db
```

look at the dataset in a web browser
```
pip install sqlite-web
sqlite_web ./sm.db
```
