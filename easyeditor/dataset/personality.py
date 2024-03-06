import json
from pathlib import Path

from torch.utils.data import Dataset
import random
import numpy as np
from ..trainer.utils import EditBatchSampler, dict_to
import torch
import sys
import typing

import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer


class PersonalityDataset(Dataset):
    """
    Dataset of PersonalityEdit.
    """
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        
        self.per_list = [
            "extraversion",
            "agreeableness", 
            "neuroticism"
        ]

        self.per2id = {
            "extraversion":0,
            "agreeableness":1, 
            "neuroticism":2
        }
        
        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 96
            
            
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
            self.tok = tokenizer
            
        with open(data_dir, "r") as f:
            raw = json.load(f)
            
        data = []
        
        self.templates = [
            "What do you think of {}?",
            "What do you feel about {}?",
            "How do you view {}?",
        ]
        for position in [
            "opinion of",
            "stance on",
            "position on",
            "attitude about",
            "view on",
            "take on",
            "impression of",
            "assessment of",
            "judgment of",
            "sentiment of",
        ]:
            self.templates.append("What is your " + position + " {}?")
        
        for case_idx, sample in enumerate(raw):
            inner_per = random.choice([0, 1, 2]) if "target_per" not in sample.keys() else self.per2id[sample["target_per"]] # 测试集generate的时候固定personality

            inner_per_text = self.per_list[inner_per] # three type of personality
        
            self.trait = inner_per_text

            inner_comp = ["Target Personailty: " + inner_per_text + "\n"]
            inner_prompt = ["Topic: " + sample["ent"] + "\n"]
            
            outer_per = ([inner_per] * len(sample[inner_per_text]))
            outer_comp = sample[inner_per_text]
            outer_temp = random.choices(self.templates, k=len(outer_per))
            outer_prompt = [t.format(sample["ent"]) for t in outer_temp]

            all_per, all_comp = [], []
        
            for idx, per in enumerate(self.per_list):
                all_per += ([idx] * len(sample[per]))
                all_comp += sample[per]
                
            all_temp = random.choices(self.templates, k=len(all_per))
            all_prompt = [t.format(sample["ent"]) for t in all_temp]
        
            data.append({
                "case_id": case_idx,
                "target_personality": inner_per_text,
                "ent": sample["ent"],
                "inner_prompt": inner_prompt,
                "inner_comp": inner_comp,
                "inner_per": inner_per,
                "outer_prompt": outer_prompt,
                "outer_comp": outer_comp,
                "all_prompt": all_prompt,
                "all_per": all_per,
                "all_comp": all_comp,
            })
            
        if size is not None:
            data = data[:size]
        self._data = data
        
        
    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
    
    def get_edit_labels(self, ids, prompts=None):
        labels = ids.clone()
        labels[labels == self.tok.pad_token_id] = -100
        return labels
    
    def _collate_fn_gpt(self, batch):
        
        inner_prompt = [prompt for b in batch for prompt in b["inner_prompt"]]
        inner_comp = [comp for b in batch for comp in b["inner_comp"]]
        outer_prompt = [prompt for b in batch for prompt in b["outer_prompt"]]
        outer_comp = [comp for b in batch for comp in b["outer_comp"]]
        all_prompt = [prompt for b in batch for prompt in b["all_prompt"]]
        all_comp = [comp for b in batch for comp in b["all_comp"]]
        
        # print("inner_prompt:", inner_prompt)
        # print("inner_comp:", inner_comp)
        # print("outer_prompt:", outer_prompt)
        # print("outer_comp:", outer_comp)
        # print("all_prompt:", all_prompt)
        # print("all_comp:", all_comp)
        
       # inner_qa = [ "Exhibit the trait of {Target Personality} when expressing opinion on the cetarin {Edit Topic}, while maintaining the expression on other topics." + q + " </s> " + a for q, a in zip(inner_prompt, inner_comp)]
        outer_qa = [ "Question: " + q + "\n </s> Answer: " + a for q, a in zip(outer_prompt, outer_comp)]
        all_qa = [ "Question: " + q + " \n </s> Answer: " + a for q, a in zip(all_prompt, all_comp)]
        
        inner_qa = [ f"{q}  {a} " + outer_qa[0] for q, a in zip(inner_prompt, inner_comp)]

        # all_per = [s for b in batch for s in b["all_per"]]
        # inner_per = [b["inner_per"] for b in batch for s in b["all_per"]]
        
        # print("all_per:", all_per)
        # print("inner_per:", inner_per)
        # outer_q = []
        # for i, q in enumerate(all_prompt):
        #     if all_per[i] == inner_per[i]:
        #         outer_q.append(q)
        
        # print("len(outer_q):", len(outer_q))
        
        try:
            batches = {
                f"{k1}_{k2}": v2
                for k1, v1 in {
                    "inner_qa": inner_qa,
                    "outer_qa": outer_qa,
                    "all_qa": all_qa,
                    "outer_q": outer_prompt,
                }.items()
                for k2, v2 in self.tok(
                    v1,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                ).items()
            }
        except Exception as e:
            print(e)
            print("inner_qa:", inner_qa)
            print("outer_qa:", outer_qa)
            print("all_qa:", all_qa)
            sys.exit(0)
        
        
        for key in ["inner_qa", "outer_qa", "all_qa"]:
            value = batches[f"{key}_input_ids"]
            mask = [([True] * value.shape[-1])] * value.shape[0]
            for i in range(value.shape[0]):
                sep_idx = list(value[i]).index(self.tok.convert_tokens_to_ids("</s>"))
                for j in range(sep_idx): #连带</s>一块mask掉
                    mask[i][j] = False
            batches[key + "_q_mask"] = mask 
                    

        batches["all_per"] = [s for b in batch for s in b["all_per"]]
        batches["inner_per"] = [b["inner_per"] for b in batch for s in b["all_per"]]
        batches["raw"] = batch

        pos_pairs = []
        for idx, b in enumerate(batch):
            for _ in range(len(b["all_prompt"])):
                pos_pairs.append([len(pos_pairs), idx])

        batches["pos_pairs"] = torch.LongTensor(pos_pairs)
        
        return batches
    
    def collate_gpt_fn(self, batch):
        
        def get_loc_idx(edit_idx):
            return (edit_idx + 1) % self.__len__()
        
        edit_idx = [mention["case_id"] for mention in batch]
        loc_idx = [get_loc_idx(mention["case_id"]) for mention in batch]

        
        edit_toks = self._collate_fn_gpt([self.__getitem__(edit_id) for edit_id in edit_idx])
        loc_toks = self._collate_fn_gpt([self.__getitem__(loc_id) for loc_id in loc_idx])
                
        edit_inner = {
            "input_ids": edit_toks["inner_qa_input_ids"],
            "attention_mask": edit_toks["inner_qa_attention_mask"],
            "labels": self.get_edit_labels(edit_toks["inner_qa_input_ids"]),
            # "q_mask": edit_toks["inner_qa_q_mask"]
        }
                
        edit_inner_q = {
            "input_ids": edit_toks["outer_q_input_ids"],
            "attention_mask": edit_toks["outer_q_attention_mask"],
            "labels": self.get_edit_labels(edit_toks["outer_q_input_ids"]),
        }
                
        edit_outer = {
            "input_ids": edit_toks["all_qa_input_ids"],
            "attention_mask": edit_toks["all_qa_attention_mask"],
            "labels": self.get_edit_labels(edit_toks["all_qa_input_ids"]),
            "q_mask": torch.tensor(edit_toks["all_qa_q_mask"], device=self.config.device)
        }

        loc = {
            "input_ids": loc_toks["all_qa_input_ids"],
            "attention_mask": loc_toks["all_qa_attention_mask"],
            "labels": self.get_edit_labels(loc_toks["all_qa_input_ids"]),
            "q_mask": torch.tensor(loc_toks["all_qa_q_mask"], device=self.config.device)
        }
        
        same_mask = torch.tensor([i == o for i, o in zip(edit_toks["inner_per"], edit_toks["all_per"])], device=self.config.device)
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "outer_per": edit_toks["all_per"],
            "inner_per": edit_toks["inner_per"],
            "trait": self.trait,
            "inner_q": edit_inner_q,
            "loc": loc,
            "cond": edit_inner,
            "same_mask": same_mask, # for computing es
            "kl_mask": loc["q_mask"] # for computing dd 
        }

        return dict_to(batch, self.config.device)