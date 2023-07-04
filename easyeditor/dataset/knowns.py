import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/known_1000.json"


class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        data_dir = Path(data_dir)
        known_loc = data_dir / "known_1000.json"
        if not known_loc.exists():
            print(f"{known_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, known_loc)

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
