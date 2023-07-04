import copy
import json
import logging
import random

from datasets import load_dataset
from torch.utils.data import Dataset
from utils import EditBatchSampler, dict_to, scr

LOG = logging.getLogger(__name__)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def filter_text(iterator):
    valid = []
    for text in iterator:
        if len(text.split(" ")) < 50:
            continue
        if not is_ascii(text):
            continue
        valid.append(text)

    return valid


class GenDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        config,
        edit_path: str,
        pct: int = 10,
        max_length: int = 200,
    ):
        version = "wikitext-103-raw-v1"
        split_str = f"{split}[:{pct}%]" if split == "train" else split
        LOG.info(f"Loading wikitext version {version}, split {split_str}")
        base_samples = load_dataset(
            "wikitext", version, cache_dir=scr(), split=split_str
        )["text"]
        self.base_samples = filter_text(base_samples)
        with open(edit_path + split[:5] + ".json", "r") as f:
            self.edit_samples = json.load(f)

        self.tok = tokenizer
        self.config = config
        self.max_length = max_length
        self.n_tokens = self.edit_samples["n_tokens"]

        len_base = len(self.base_samples)
        len_edit = len(self.edit_samples["original"])
        LOG.info(f"Loaded {len_base} wiki-103 samples and {len_edit} edit samples")

        if config.data.wiki_webtext:
            self.use_wiki = True
            LOG.info("** Using webtext for wiki base samples **")
            webtext = load_dataset(
                "stas/openwebtext-10k", split="train", cache_dir=scr()
            )["text"]
            n_train = int(len(webtext) * 0.9)
            if split == "train":
                self.base_samples = webtext[:n_train]
            else:
                self.base_samples = webtext[n_train:]
        else:
            self.use_wiki = False

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(
            n,
            memorize_mode=self.config.single_batch,
            loc_disjoint=not self.use_wiki,
            seed=self.config.seed,
        )
        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)

            edit_batch = [self.edit_samples["completions"][idx] for idx in edit_idxs]
            loc_batch = [
                self.base_samples[idx % len(self.base_samples)] for idx in loc_idxs
            ]

            edit_toks = self.tok(edit_batch, padding=True, return_tensors="pt")
            loc_toks = self.tok(
                loc_batch,
                padding=True,
                return_tensors="pt",
                truncation=self.config.data.wiki_webtext,
                max_length=self.max_length,
            )

            edit_inner = {**edit_toks}
            edit_inner["labels"] = self.get_edit_labels(edit_toks["input_ids"])

            edit_outer = copy.deepcopy(edit_inner)
            if self.config.data.rephrase:
                lens = (edit_outer["input_ids"] != -100).sum(-1)
                remove = random.randint(0, (min(lens) - self.n_tokens) // 2)
                for k, v in edit_outer.items():
                    edit_outer[k] = v[:, remove:]

            loc = {**loc_toks}
            loc["labels"] = self.get_labels(loc_toks["input_ids"])
            cond = {**edit_toks}

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond,
            }

            yield dict_to(batch, self.config.device)

    def __len__(self):
        return len(self.edit_samples["original"])

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported for GPT2")

    def get_edit_labels(self, ids):
        self._check_padding(ids)

        labels = ids.clone()
        end_idxs = (labels != self.tok.pad_token_id).sum(-1)
        for batch_idx, end_idx in enumerate(end_idxs):
            labels[batch_idx, : end_idx - self.n_tokens] = -100
        labels[labels == self.tok.pad_token_id] = -100
        return labels

    def get_labels(self, ids):
        self._check_padding(ids)

        return ids.masked_fill(ids == self.tok.pad_token_id, -100)

    def __getitem__(self, idx):
        return self.base_samples[idx]
