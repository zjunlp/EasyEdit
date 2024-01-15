import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to
from typing import Dict, List, Any, Optional


# This class is only for SERAC, MEND, FT, LoRA in training stage
class SanitizationTrainDataset(Dataset):
    
    # 暂时1:1吧
    
    def generate_next_locality_index(self):
        if self.locality_index >= len(self.origin_data["K_R"]):
            self.locality_index = 0
        self.locality_index += 1
        return self.locality_index - 1

    def __init__(
        self, 
        data_dir: str,
        template: str, 
        specify_answers: str=None,         # 如果选定了，那么每次都只对选定的answer可见，其余都不可见
        size: Optional[int] = None, 
        config=None, 
        *args, 
        **kwargs
    ):
        assert "train" in data_dir and "test" not in data_dir
        data_dir = Path(data_dir)
        st_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 40

        # For Meta Training
        if config is not None and hasattr(config, 'tokenizer_name'):
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

        with open(st_loc, "r") as f:
            data:dict = json.load(f)

        item_template: dict = {
            "prompt": None,
            "target_new": None,
            "ground_truth": None,
            "locality_prompt": None,
            "locality_ground_truth": None
        }

        # 根据data拿出answer
        answers = list(set([item["ground_truth"].lower() for item in data['K_F']]))
        assert len(answers) == 5

        locality_idx_start = -1
        if specify_answers is not None:
            # 表明不是对全部进行，而是只拿特定的
            assert specify_answers in answers, f"`{specify_answers}` is not in `{answers}`"
            locality_idx_start = answers.index(specify_answers)
            tmp = []
            for item in data["K_F"]:
                if item["ground_truth"].lower() == specify_answers:
                    tmp.append(item)
            assert len(tmp) == 16, f"{len(tmp)} != 16"
            data["K_F"] = tmp
            # 取K_R
            # 比如idx为1的话，理论上应该是[80:160]
            proportion = {0:[0,90],1:[90,180],2:[180,270],3:[270,360],4:[360,453]}[locality_idx_start]
            data["K_R"] = data["K_R"][proportion[0]:proportion[1]]
        
        self.locality_index = 0
        self.origin_data = data
        self.data = []
        for i in range(len(self.origin_data["K_F"])):
            cur_item = self.origin_data["K_F"][i]
            cur_retain_item = self.origin_data["K_R"][self.generate_next_locality_index()]
            self.locality_index += 1
            self.data.append({
                "prompt": template.format(cur_item["question"]),
                "target_new": cur_item["target_new"],
                "ground_truth": cur_item["ground_truth"],
                "locality_prompt": template.format(cur_retain_item["question"]),
                "locality_ground_truth": cur_retain_item["ground_truth"]
            })

        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded dataset with {len(self)} elements")

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)


    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = ["{} >> {} || {}".format(b['ground_truth'],
                                        b["target_new"],
                                        b['prompt']) for b in batch]
        # no rephrase_prompt
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

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
        cond = ["{} >> {} || {}".format(b['ground_truth'],
                                        b["target_new"],
                                        b['prompt']) for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

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

        cond = {k[5:]: v for k, v in batches.items() if k.startswith("cond")}
        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
