import random

import torch
import numpy as np
import audiotools as at
from audiotools import AudioSignal
import pandas as pd

import soundmaterial as sm

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df: pd.DataFrame, 
        sample_rate: int = 44100,
        n_samples: int = 44100,
        num_channels: int = 1,
        audio_key: str = "path",
        aux_keys: list = [],
        transform=None
    ):
        self.df = df
        self.transform = transform

        self.audio_key = audio_key
        self.aux_keys = aux_keys
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.num_channels = num_channels

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        import time
        t0 = time.time()
        row = self.df.iloc[idx]
        # check if the duration is in the row
        if "duration" in row:
            total_duration = row["duration"]
            duration = self.n_samples / self.sample_rate
            # sample a random offset 
            state = at.util.random_state(None)
            lower_bound = 0 
            upper_bound = max(total_duration - duration, 0)
            offset = state.uniform(lower_bound, upper_bound)
            sig = AudioSignal(
                row[self.audio_key], 
                duration=duration, 
                offset=offset,
                sample_rate=self.sample_rate
            )
        else:
            sig = AudioSignal.excerpt(
                row[self.audio_key], 
                duration=self.n_samples/self.sample_rate, 
                sample_rate=self.sample_rate
            )
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

        if self.transform:
            sig.samples = self.transform(sig.samples)

        out = {"wav": sig.samples[0], "sample_rate": sig.sample_rate}
        for key in self.aux_keys:
            out[key] = row[key]
        
        # print(f"{idx} took: {time.time() - t0}")
        return out

    @staticmethod
    def collate(batch):
        wavs = [item["wav"] for item in batch]
        wav = torch.stack(wavs)

        sample_item = batch[0]
        keys = list(sample_item.keys())
        out = {}
        out["wav"] = wav
        for item in batch:
            assert list(item.keys()) == keys
            for key in keys:
                if key == "wav":
                    continue
                if key not in out:
                    out[key] = []
                out[key].append(item[key])
        
        for key in out:
            if isinstance(out[key][0], torch.Tensor):
                out[key] = torch.stack(out[key])
        
        return out

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