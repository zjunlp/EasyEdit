import logging
import random

import jsonlines
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartTokenizerFast
from utils import EditBatchSampler, dict_to

LOG = logging.getLogger(__name__)


class Seq2SeqAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tok = tokenizer
        self.data = []
        self.config = config

        def extract(d):
            ex = {
                k: d[k]
                for k in [
                    "input",
                    "prediction",
                    "alternatives",
                    "filtered_rephrases",
                    "output",
                ]
            }
            if ex["input"] in ex["filtered_rephrases"]:
                ex["filtered_rephrases"].remove(ex["input"])
            return ex

        with jsonlines.open(data_path) as f:
            for d in f:
                extracted = extract(d)
                if (
                    len(extracted["alternatives"]) > 0
                    and len(extracted["filtered_rephrases"]) > 0
                ):
                    self.data.append(extracted)

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view
        if self.config.data.zsre_nq and "train" not in data_path:
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset

            self.nq = NQDataset(
                self.config.data.nq_path
                + ("/train.json" if "train" in data_path else "/validation.json"),
                tokenizer,
                config,
            )
        else:
            self.use_nq = False

    def is_bart(self):
        return isinstance(self.tok, BartTokenizer) or isinstance(
            self.tok, BartTokenizerFast
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        new_label = random.choice(self.data[item]["alternatives"])
        rephrase = random.choice(self.data[item]["filtered_rephrases"])
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"],
            "rephrase": rephrase,
            "alt": new_label,
            "answers": [x["answer"] for x in self.data[item]["output"]],
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                new_label,
                self.data[item]["input"],
            ),
        }

        return output

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        ne = self.config.data.n_edits
        trg = [b["answers"][0] for b in batch[:-ne]] + [b["alt"] for b in batch[-ne:]]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": [b["cond"] for b in batch[-ne:]],
                "rephrase": [b["rephrase"] for b in batch[-ne:]],
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        if self.is_bart():  # For consistency with Cao et al
            batches["trg_input_ids"][:, 0] = self.tok.eos_token_id
        batches["raw"] = batch
        return batches

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported")

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(
            n,
            memorize_mode=self.config.single_batch,
            loc_disjoint=not self.use_nq,
            seed=self.config.seed,
        )

        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in idxs])

            ne = self.config.data.n_edits
            edit_decoder_inputs = toks["trg_input_ids"][-ne:]
            edit_labels = self.get_edit_labels(edit_decoder_inputs)
            edit_attention_mask = toks["trg_attention_mask"][-ne:]

            edit_inner = {}
            edit_inner["input_ids"] = toks["src_input_ids"][-ne:]
            edit_inner["attention_mask"] = toks["src_attention_mask"][-ne:]
            if self.is_bart():
                edit_inner["decoder_input_ids"] = edit_decoder_inputs
                edit_inner["decoder_attention_mask"] = edit_attention_mask
            edit_inner["labels"] = edit_labels

            if self.config.data.rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["rephrase_input_ids"]
                edit_outer["attention_mask"] = toks["rephrase_attention_mask"]
                if self.is_bart():
                    edit_outer["decoder_input_ids"] = edit_decoder_inputs
                    edit_outer["decoder_attention_mask"] = edit_attention_mask
                edit_outer["labels"] = edit_labels
            else:
                edit_outer = edit_inner

            loc = {}
            if self.use_nq:
                batch = [self.nq[idx] for idx in loc_idxs]
                questions = [b[0] for b in batch]
                answers = [b[1] for b in batch]
                loc = dict(
                    self.tok(
                        questions,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.max_length,
                        truncation=True,
                    )
                )
                trg_toks = dict(
                    self.tok(
                        answers,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.max_length,
                        truncation=True,
                    )
                )
                if self.is_bart():
                    trg_toks["input_ids"][:, 0] = self.tok.eos_token_id
                    loc["decoder_input_ids"] = trg_toks["input_ids"]
                loc["decoder_attention_mask"] = trg_toks["attention_mask"]
                loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])
            else:
                loc["input_ids"] = toks["src_input_ids"][:-ne]
                loc["attention_mask"] = toks["src_attention_mask"][:-ne]
                if self.is_bart():
                    loc["decoder_input_ids"] = toks["trg_input_ids"][:-ne]
                loc["decoder_attention_mask"] = toks["trg_attention_mask"][:-ne]
                loc["labels"] = self.get_edit_labels(toks["trg_input_ids"][:-ne])

            cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond,
                "raw": toks["raw"],
            }

            yield dict_to(batch, self.config.device)
