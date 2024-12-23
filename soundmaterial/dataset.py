import random

import torch
import numpy as np
import pandas as pd

import soundmaterial as sm
import vampnet.signal as sn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
        df: pd.DataFrame, 
        sample_rate: int = 44100,
        n_samples: int = 44100,
        num_channels: int = 1,
        audio_key: str = "path",
        aux_keys: list = [],
        transform=None, 
        max_examples: int = None,
        use_chunk_table: bool = False 
    ):
        self.df = df
        self.transform = transform

        self.audio_key = audio_key
        self.aux_keys = aux_keys
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.num_channels = num_channels
        assert self.num_channels in [1, 2], f"multichannel not supported yet."

        self.use_chunk_table = use_chunk_table

        # what's the total duration of the dataset in samples? 
        # we need this to sample random excerpts
        self.total_samples = df["duration"].sum() * sample_rate
        self.total_excerpts = int(self.total_samples / n_samples)
        print(f"total excerpts: {self.total_excerpts}")
        if max_examples is not None:
            print(f"limiting to {max_examples} examples")
            self.total_excerpts = min(max_examples, self.total_excerpts)


    def __len__(self):
        return self.total_excerpts
    
    def __getitem__(self, idx):
        import time
        t0 = time.time()
        row = self.df.iloc[idx % len(self.df)]
        # check if the duration is in the row
        state = sn.random_state(idx)
        if "duration" and "offset" in row and self.use_chunk_table:
            # print("using chunk table")
            duration = row["duration"]
            offset = row["offset"]
            
            # nudge a little back if we're too close to the end
            if offset > 0 and (duration + offset) > row["total_duration"]:
                offset = row["total_duration"] - duration
                if offset < 0: # we tried
                    offset = 0
            
            # assert chunk is big enough 
            assert duration >= self.n_samples / self.sample_rate, f"chunk too small: {duration}"
            sig = sn.read_from_file(
                row[self.audio_key], 
                duration=duration, 
                offset=offset,
            )
            sig = sn.resample(sig, self.sample_rate)
        elif "duration" in row:
            total_duration = row["duration"]
            duration = self.n_samples / self.sample_rate
            # sample a random offset 
            lower_bound = 0 
            upper_bound = max(total_duration - duration, 0)
            offset = state.uniform(lower_bound, upper_bound)
            sig = sn.read_from_file(
                row[self.audio_key], 
                duration=duration, 
                offset=offset,
            )
            sig = sn.resample(sig, self.sample_rate)
        else:
            sig = sn.excerpt(
                row[self.audio_key], 
                duration=self.n_samples/self.sample_rate, 
                sample_rate=self.sample_rate, 
                state=sn.random_state(idx)
            )
        # pad up to the desired duration
        if sig.num_samples < self.n_samples:
            num_pad = self.n_samples - sig.num_samples
            sig.wav = torch.nn.functional.pad(sig.wav, (0, num_pad))

        if self.num_channels == 1:
            sig = sn.to_mono(sig)
        elif self.num_channels == 2:
            if sig.num_channels == 1:
                sig.wav = torch.cat([sig.wav, sig.wav], dim=1)
            elif sig.num_channels == 2:
                pass
            elif sig.num_channels > 2:
                sig.wav = sig.wav[:, :2]
        else:
            raise ValueError(f"Invalid number of channels: {self.num_channels}")

        if self.transform:
            sig = self.transform(sig)
    
        out = {"sig": sig }
        for key in self.aux_keys:
            out[key] = row[key]


        return out

    @staticmethod
    def collate(batch):
        return sn.collate(batch)


class BalancedWeightedDataset:

    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.weights = weights
        if self.weights is None:
            self.weights = [1 for _ in range(len(datasets))]
        
        self.total_len = sum([len(d) for d in datasets])    
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        return self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])]


def train_test_split(df, test_size=0.1, seed=42):
    print(f"splitting dataset with test_size={test_size}, seed={seed}")
    np.random.seed(seed)
    n = len(df)
    test_n = int(n * test_size)
    test_idxs = np.random.choice(n, test_n, replace=False)
    # train_idxs = np.array([i for i in range(n) if i not in test_idxs])
    train_idxs = np.array(list(set(range(n)) - set(test_idxs)))
    print(f"train: {len(train_idxs)}, test: {len(test_idxs)}")
    return df.iloc[train_idxs], df.iloc[test_idxs]

if __name__ == "__main__":

    conn = sm.connect("./sm.db")
    query = "SELECT * FROM audio_file"
    
    df = pd.read_sql_query(query, conn)
    tfd, vdf = train_test_split(df)
    train_dataset = AudioDataset(df)
    val_dataset = AudioDataset(vdf)


    print(len(train_data), train_data[0])
    print(len(val_data), val_data[0])

    breakpoint()