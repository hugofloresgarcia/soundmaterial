# soundmaterial
a python library for examining (with your eyes and ears) large sound datasets with a playful curiosity 

## setup
```bash
git clone https://github.com/hugofloresgarcia/soundmaterial
cd soundmaterial
pip install -e .
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

look at the dataset in a web browser
```
pip install sqlite-web
sqlite_web ./sm.db
```
