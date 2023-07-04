import collections
import json
import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

PARAREL_RELATION_NAMES = [
    "P39",
    "P264",
    "P37",
    "P108",
    "P131",
    "P103",
    "P176",
    "P30",
    "P178",
    "P138",
    "P47",
    "P17",
    "P413",
    "P27",
    "P463",
    "P364",
    "P495",
    "P449",
    "P20",
    "P1376",
    "P1001",
    "P361",
    "P36",
    "P1303",
    "P530",
    "P19",
    "P190",
    "P740",
    "P136",
    "P127",
    "P1412",
    "P407",
    "P140",
    "P279",
    "P276",
    "P159",
    "P106",
    "P101",
    "P937",
]


def pararel(data_path: str = "datasets/pararel.json"):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        PARAREL = collections.defaultdict(dict)
        # download relations from github
        for r in tqdm(PARAREL_RELATION_NAMES, "downloading pararel data"):
            with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/yanaiela/pararel/main/data/pattern_data/graphs_json/{r}.jsonl"
            ) as url:
                graphs = [
                    json.loads(d.strip()) for d in url.read().decode().split("\n") if d
                ]
                PARAREL[r]["graphs"] = graphs
            with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/yanaiela/pararel/main/data/trex_lms_vocab/{r}.jsonl"
            ) as url:
                vocab = [
                    json.loads(d.strip()) for d in url.read().decode().split("\n") if d
                ]
                PARAREL[r]["vocab"] = vocab
        with open(data_path, "w") as f:
            json.dump(PARAREL, f)
        return PARAREL


def pararel_expanded(
    data_path: str = "datasets/pararel_expanded.json", obj_label_replacement=None
):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        PARAREL = pararel()
        PARAREL_EXPANDED = collections.defaultdict(dict)
        # expand relations into sentences, grouped by their uuid
        for key, value in tqdm(
            PARAREL.items(), "expanding pararel dataset into full sentences"
        ):
            for vocab in value["vocab"]:
                for graph in value["graphs"]:
                    if not PARAREL_EXPANDED.get(vocab["uuid"]):
                        PARAREL_EXPANDED[vocab["uuid"]] = {
                            "sentences": [],
                            "relation_name": key,
                            "obj_label": vocab["obj_label"],
                        }
                    sentence = graph["pattern"]
                    full_sentence = sentence.replace("[X]", vocab["sub_label"]).replace(
                        "[Y]", "[MASK]"
                    )
                    PARAREL_EXPANDED[vocab["uuid"]]["sentences"].append(full_sentence)
        with open(data_path, "w") as f:
            json.dump(PARAREL_EXPANDED, f)
        return PARAREL_EXPANDED
