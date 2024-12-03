import random

import torch
import numpy as np
import audiotools as at
from audiotools import AudioSignal
import pandas as pd

import soundmaterial as sm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
        df: pd.DataFrame, 
        sample_rate: int = 44100,
        n_samples: int = 44100,
        num_channels: int = 1,
        audio_key: str = "path",
        aux_keys: list = [],
        transform=None, 
        max_examples: int = None
    ):
        self.df = df
        self.transform = transform

        self.audio_key = audio_key
        self.aux_keys = aux_keys
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.num_channels = num_channels
        assert self.num_channels in [1, 2], f"multichannel not supported yet."

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
        if "duration" in row:
            total_duration = row["duration"]
            duration = self.n_samples / self.sample_rate
            # sample a random offset 
            state = at.util.random_state(idx)
            lower_bound = 0 
            upper_bound = max(total_duration - duration, 0)
            offset = state.uniform(lower_bound, upper_bound)
            sig = AudioSignal(
                row[self.audio_key], 
                duration=duration, 
                offset=offset,
                sample_rate=self.sample_rate
            )
            sig.metadata["load_time"] = time.time() - t0
        else:
            sig = AudioSignal.excerpt(
                row[self.audio_key], 
                duration=self.n_samples/self.sample_rate, 
                sample_rate=self.sample_rate
            )
        sig = sig.resample(self.sample_rate)
        # pad up to the desired duration
        # if sig.duration < self.duration:
        #     num_pad = int((self.duration - sig.duration) * self.sample_rate)
        #     sig.samples = torch.nn.functional.pad(sig.samples, (0, num_pad), mode="reflect")
        if sig.length < self.n_samples:
            num_pad = self.n_samples - sig.length
            sig.samples = torch.nn.functional.pad(sig.samples, (0, num_pad))

        if self.num_channels == 1:
            sig = sig.to_mono()
        elif self.num_channels == 2:
            if sig.num_channels == 1:
                sig.samples = torch.cat([sig.samples, sig.samples], dim=1)
            elif sig.num_channels == 2:
                pass
            elif sig.num_channels > 2:
                sig.samples = sig.samples[:, :2]
        else:
            raise ValueError(f"Invalid number of channels: {self.num_channels}")

        out = {"wav": sig.samples[0], "sample_rate": sig.sample_rate}
        out["signal"] = sig
        if self.transform is not None:
            out["transform_args"] = self.transform.instantiate(state, signal=sig)

        for key in self.aux_keys:
            out[key] = row[key]
        
        # print(f"{idx} took: {time.time() - t0}")
        return out

    @staticmethod
    def collate(batch):
        return at.util.collate(batch)

def train_test_split(df, test_size=0.1, seed=42):
    np.random.seed(seed)
    n = len(df)
    test_n = int(n * test_size)
    test_idxs = np.random.choice(n, test_n, replace=False)
    train_idxs = np.array([i for i in range(n) if i not in test_idxs])
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