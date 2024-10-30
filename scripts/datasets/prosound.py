
from soundmaterial.create import create_db
from soundmaterial.add import add_dataset
from soundmaterial.subset import create_subset

path_to_prosound = "/media/pancho/prosound_core_complete/Anns Animals"
db_path = "prosound.db"
dataset_name = "prosound"

# create a database
create_db(db_path)

add_dataset(
    db_path, 
    path_to_prosound, 
    dataset_name
)

# TODO (layton): 
# here, we would add the captions to the caption table inthe database



# TODO (layton):
# here, instead of making a single subset, we would make multiple subsets
# for different classes of sounds (e.g. speech, machines, animals, musical instruments, synthesizers, etc.) for data balancing

# now, let's create a subset for training
# for our subset, let's only keep the files 
# whose duration is shorter than 5 minutes (for dataloading speed purposes)
query = "SELECT * FROM audio_file WHERE duration < 300 and format = 'wav'"
output_folder = "data/prosound_training_subset_symlinks"
create_subset(
    db_path,
    output_folder,
    query,
    symlinks=True # don't copy the files, just symlink!
)
