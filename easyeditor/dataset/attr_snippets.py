import collections
import json
from pathlib import Path

import torch

from ..util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/attribute_snippets.json"


class AttributeSnippets:
    """
    Contains wikipedia snippets discussing entities that have some property.

    More formally, given a tuple t = (s, r, o):
    - Let snips = AttributeSnippets(DATA_DIR)
    - snips[r][o] is a list of wikipedia articles for all s' such that t' = (s', r, o) is valid.
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        snips_loc = data_dir / "attribute_snippets.json"
        if not snips_loc.exists():
            print(f"{snips_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, snips_loc)

        with open(snips_loc, "r") as f:
            snippets_list = json.load(f)

        snips = collections.defaultdict(lambda: collections.defaultdict(list))

        for el in snippets_list:
            rid, tid = el["relation_id"], el["target_id"]
            for sample in el["samples"]:
                snips[rid][tid].append(sample)

        self._data = snips
        self.snippets_list = snippets_list

    def __getitem__(self, item):
        return self._data[item]
