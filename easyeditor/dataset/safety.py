import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer
from ..util.globals import *
from ..trainer.utils import dict_to

class SafetyDataset(Dataset):
    """
    Dataset of SafeEdit
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        unsafety_loc = data_dir

        if(config is not None):
            self.config = config
        if(config is not None and hasattr(config, 'max_length')):
            self.max_length = config.max_length
        else:
            self.max_length = 1000

        if(config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name
            )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        with open(unsafety_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            data.append(
                    {
                        "case_id": record["id"],
                        "category": record["unsafety category"],
                        "question": record["question"],
                        "prompt": record["adversarial prompt"],
                        "target_new": record["safe generation"],
                        "ground_truth": record["unsafe generation"],  
                        "locality_prompt": record["knowledge constrain"]["prompt"],
                        "locality_ground_truth": record["knowledge constrain"]["answer"],
                        "cond": "{} >> {} || {}".format(
                            record["unsafe generation"],
                            record["safe generation"],
                            record["adversarial prompt"],
                        ),
                    }
                )
            if "generalization test" in record.keys():
                data[-1]["general_prompt"] = [
                                        record["generalization test"]["test input of only harmful question"],
                                        record["generalization test"]["test input of other attack prompt input"],
                                        record["generalization test"]["test input of other question input"],
                                        record["generalization test"]["test input of other questions and attack prompts"],
                                        ]

        if size is not None:
            data = data[:size]
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = [b["cond"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        
        src = [src_ + trg_ for src_, trg_ in zip(src, trg)]  
        loc = [loc_ + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }


        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"]) 

        edit_inner["labels"] = edit_labels   



        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        cond = {k[5:]: v for k, v in batches.items() if k.startswith("cond")}
        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = [b["cond"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        
        src = [src_ + trg_ for src_, trg_ in zip(src, trg)]  
        loc = [loc_ + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"]) 

        edit_inner["labels"] = edit_labels   


        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        cond = {k[5:]: v for k, v in batches.items() if k.startswith("cond")}
        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            
        }
        return dict_to(batch, self.config.device)
