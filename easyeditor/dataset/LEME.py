import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class LongFormEditDataset(Dataset):
    """
    Dataset for Long-form Evaluation of Model Editing (LEME).
    Based on the paper: "Long-form evaluation of model editing"
    
    Supports two dataset types:
    - zsre: zsre_mend_eval_with_coupled_entities.json
    - counterfact: counterfact_with_coupled_entities.json (TODO)
    """

    def __init__(self, data_dir: str, dataset_type: str = "zsre", size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        longform_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 40
            
        self.dataset_type = dataset_type
        
        # For Meta Training
        if config is not None and hasattr(config, 'tokenizer_name'):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
            # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            elif 'qwen' in config.model_name.lower():
                tokenizer.eos_token='<|endoftext|>'
                tokenizer.pad_token='<|endoftext|>'
                tokenizer.unk_token='<|endoftext|>'
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            elif 'mistral' in config.model_name.lower():
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('MistralTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        with open(longform_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            if self.dataset_type == "zsre":
                processed_record = self._process_zsre_data(record, i)
            elif self.dataset_type == "counterfact":
                processed_record = self._process_counterfact_data(record, i)
            
            data.append(processed_record)

        if size is not None:
            data = data[:size]
        self._data = data

    def _process_zsre_data(self, record, case_id):
        return {
            "case_id": case_id,
            "subject": record["subject"],
            "prompt": record["src"],
            "target_new": record["alt"], 
            "ground_truth": record["answers"][0] if record["answers"] else "",
            "rephrase": record["rephrase"],
            "locality": record["loc"],
            "locality_ans": record["loc_ans"],
            "cond": record.get("cond", ""),
            # longform evaluation related fields
            "longform_subject_prompt": self._extract_subject_prompt(record),
            "longform_coupled_entities": self._extract_coupled_entities(record),
            # we set portability_metrics to None if not provided
            "portability_personas": record["personas"] if "personas" in record else None,
            "portability_hop": record["mhop"] if "mhop" in record else None,
            "portability_hop_ans":record["mhop_ans"] if "mhop_ans" in record else None,     
        }

    def _process_counterfact_data(self, record, case_id):
        rewrite = record["requested_rewrite"]
        prompt = rewrite["prompt"].format(rewrite["subject"]) + "?"
        ground_truth = rewrite["target_true"]["str"]
        target_new = rewrite["target_new"]["str"]
        return { 
            "case_id": case_id,
            "subject": rewrite["subject"],
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "rephrase": record["paraphrase_prompts"],
            "locality": record["neighborhood_prompts"],
            "locality_ans": [rewrite["target_true"]["str"]] * len(record["neighborhood_prompts"]),
            "cond": f"{ground_truth} >> {target_new} || {prompt}",
            # longform evaluation related fields
            "longform_subject_prompt": self._extract_subject_prompt(record),
            "longform_coupled_entities": self._extract_coupled_entities(record),
            # we set portability_metrics to None if not provided
            "portability_personas": record["personas"] if "personas" in record else None,
            "portability_hop": record["mhop"] if "mhop" in record else None,
            "portability_hop_ans":record["mhop_ans"] if "mhop_ans" in record else None,     
        }

    def _extract_subject_prompt(self, record):
        # TODO: Extract subject prompt for long-form evaluation
        coupled_data = record.get("coupled_prompts_and_properties", {})
        subject_entity = coupled_data.get("subject_entity", {})
        return subject_entity.get("coupled_prompt", None)

    def _extract_coupled_entities(self, record):
        # TODO: Extract coupled entities for long-form evaluation
        coupled_data = record.get("coupled_prompts_and_properties", {})
        return coupled_data.get("coupled_entities", [])

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        rephrase = [b["rephrase"] for b in batch]
        cond = [b["cond"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
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

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

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
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        rephrase = [b["rephrase"] for b in batch]
        cond = [b["cond"] for b in batch]
        
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        # if (hasattr(self.config, 'alg') and self.config.alg == 'SERAC') or \
        #         (hasattr(self.config, 'alg_name') and self.config.alg_name == 'SERAC'):
        #     def flatten(nested_list: typing.List[typing.List]):
        #         return [item for nested_list_ in nested_list for item in nested_list_]
        #
        #     trg = [' ' + trg_ for trg_ in trg]
        #     loc_ans = [' ' + loc_ans_ for loc_ans_ in loc_ans]
        #     src = [[src_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     rephrase = [[rephrase_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for rephrase_, trg_ in zip(rephrase, trg)]
        #     loc = [[loc_ + self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #     trg = [[self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     loc_ans = [[self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #
        #     src, rephrase, trg, loc, loc_ans = flatten(src), flatten(rephrase), flatten(trg), flatten(loc), flatten(loc_ans)
        #
        # else:
        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        rephrase = [rephrase_ + ' ' + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        if 'gpt' in self.config.tokenizer_class.lower():
            trg = [' ' + t for t in trg]
            loc_ans = [' ' + t for t in loc_ans]
            
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond,
                "rephrase": rephrase,
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

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

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
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
