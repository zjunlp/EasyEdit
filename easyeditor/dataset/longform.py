import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class LongFormDataset(Dataset):
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        
        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 40

        if config is not None and hasattr(config, 'tokenizer_name'):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
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
            if 'qwen' in config.model_name.lower():
                tokenizer.eos_token='<|endoftext|>'
                tokenizer.pad_token='<|endoftext|>'
                tokenizer.unk_token='<|endoftext|>'
            self.tok = tokenizer

        with open(data_dir, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            is_new_format = "subject" in record and "src" in record
            is_requested_rewrite_format = "requested_rewrite" in record
            
            data_item = {
                "subject": "",
                "prompt": "",
                "target_new": "",
                "target_true": "",
                "portability_data": [],
                "portability_answer": "",
                "portability_s": None,
                "portability_r": None,
                "portability_l": None,
                "locality_data": [],
                "locality_rs": None,
                "locality_f": None
            }
            
            if is_new_format:
                data_item["subject"] = record.get("subject", "")
                data_item["prompt"] = record.get("src", "")
                data_item["target_new"] = record.get("pred", "")
                data_item["portability_answer"] = record.get("pred", "")
                
                if "answers" in record:
                    if isinstance(record["answers"], list):
                        data_item["target_true"] = record["answers"][0] if record["answers"] else ""
                    else:
                        data_item["target_true"] = record["answers"]
                
                if "rephrase" in record:
                    if isinstance(record["rephrase"], str):
                        if record["rephrase"].startswith("- "):
                            portability_prompts = record["rephrase"].split("- ")[1:]
                        else:
                            portability_prompts = [record["rephrase"]]
                    elif isinstance(record["rephrase"], list):
                        portability_prompts = record["rephrase"]
                        
                    data_item["portability_s"] = []
                    for prompt in portability_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                    
                    data_item["portability_data"] = portability_prompts
                
                if "loc" in record and record["loc"]:
                    locality_prompt = record["loc"]
                    locality_answer = record.get("loc_ans", data_item["target_true"])
                    
                    data_item["locality_rs"] = [{
                        "prompt": locality_prompt,
                        "ground_truth": locality_answer
                    }]
                    
                    data_item["locality_data"] = [{
                        "prompt": locality_prompt,
                        "subject": data_item["subject"],
                        "target": locality_answer
                    }]
                
            elif is_requested_rewrite_format:
                rewrite = record["requested_rewrite"]
                
                data_item["subject"] = rewrite.get("subject", "")
                
                prompt = rewrite.get("prompt", "")
                if prompt and data_item["subject"] and "{}" in prompt:
                    data_item["prompt"] = prompt.format(data_item["subject"])
                else:
                    data_item["prompt"] = rewrite.get("prompt_full", prompt)
                
                data_item["target_new"] = rewrite.get("target_new", {}).get("str", "")
                data_item["target_true"] = rewrite.get("target_true", {}).get("str", "")
                data_item["portability_answer"] = data_item["target_new"]
                
                if "paraphrase_prompts" in record and record["paraphrase_prompts"]:
                    paraphrase_prompts = record["paraphrase_prompts"]
                    data_item["portability_data"] = paraphrase_prompts
                    
                    data_item["portability_s"] = []
                    for prompt in paraphrase_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                
                if "neighborhood_prompts" in record and record["neighborhood_prompts"]:
                    neighborhood_prompts = record["neighborhood_prompts"]
                    
                    data_item["locality_rs"] = []
                    data_item["locality_data"] = []
                    
                    for i, prompt in enumerate(neighborhood_prompts):
                        answer = data_item["target_true"]
                        data_item["locality_rs"].append({
                            "prompt": prompt,
                            "ground_truth": answer
                        })
                        data_item["locality_data"].append({
                            "prompt": prompt,
                            "subject": data_item["subject"],
                            "target": answer
                        })
                
                if "coupled_prompts_and_properties" in record:
                    data_item["coupled_prompts_and_properties"] = record["coupled_prompts_and_properties"]
                
            else:
                data_item["subject"] = record.get("subject", "")
                prompt = record.get("prompt", record.get("text", ""))
                
                if data_item["subject"] and prompt and "{}" in prompt:
                    data_item["prompt"] = prompt.format(data_item["subject"])
                else:
                    data_item["prompt"] = record.get("prompt_full", prompt)
                
                data_item["target_new"] = record.get("target_new", record.get("pred", ""))
                data_item["target_true"] = record.get("target_true", record.get("ground_truth", ""))
                data_item["portability_answer"] = data_item["target_new"]
                
                if not data_item["target_true"] and "answers" in record:
                    if isinstance(record["answers"], list) and record["answers"]:
                        data_item["target_true"] = record["answers"][0]
                    elif isinstance(record["answers"], str):
                        data_item["target_true"] = record["answers"]
                
                portability_prompts = []
                if "paraphrase_prompts" in record and record["paraphrase_prompts"]:
                    portability_prompts = record["paraphrase_prompts"]
                elif "rephrase" in record:
                    if isinstance(record["rephrase"], str):
                        if record["rephrase"].startswith("- "):
                            portability_prompts = record["rephrase"].split("- ")[1:]
                        else:
                            portability_prompts = [record["rephrase"]]
                    elif isinstance(record["rephrase"], list):
                        portability_prompts = record["rephrase"]
                
                if portability_prompts:
                    data_item["portability_data"] = portability_prompts
                    data_item["portability_s"] = []
                    for prompt in portability_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                
                locality_prompts = []
                if "neighborhood_prompts" in record and record["neighborhood_prompts"]:
                    locality_prompts = record["neighborhood_prompts"]
                elif "loc" in record and record["loc"]:
                    locality_prompts = [record["loc"]]
                
                if locality_prompts:
                    data_item["locality_rs"] = []
                    data_item["locality_data"] = []
                    
                    for i, prompt in enumerate(locality_prompts):
                        answer = ""
                        if "neighborhood_answers" in record and len(record["neighborhood_answers"]) > i:
                            answer = record["neighborhood_answers"][i]
                        elif "loc_ans" in record and i == 0:
                            answer = record["loc_ans"]
                        else:
                            answer = data_item["target_true"]
                        
                        data_item["locality_rs"].append({
                            "prompt": prompt,
                            "ground_truth": answer
                        })
                        data_item["locality_data"].append({
                            "prompt": prompt,
                            "subject": data_item["subject"],
                            "target": answer
                        })
            
            if "coupled_prompts_and_properties" in record:
                data_item["coupled_prompts_and_properties"] = record["coupled_prompts_and_properties"]
            
            if data_item["subject"] and data_item["prompt"] and data_item["target_new"]:
                data.append(data_item)

        if size is not None:
            data = data[:size]
        self._data = data

        print(f"\n===== 数据集统计 =====")
        print(f"总样本数: {len(data)}")
        
        has_portability = sum(1 for item in data if item['portability_data'] and len(item['portability_data']) > 0)
        has_portability_s = sum(1 for item in data if item['portability_s'] and len(item['portability_s']) > 0)
        has_locality = sum(1 for item in data if item['locality_data'] and len(item['locality_data']) > 0)
        has_locality_rs = sum(1 for item in data if item['locality_rs'] and len(item['locality_rs']) > 0)
        
        print(f"有迁移性数据的样本数: {has_portability}")
        print(f"有局部性数据的样本数: {has_locality}")
        
        if len(data) > 0:
            print("\n===== 样本示例 =====")
            sample = data[0]
            print(f"Subject: {sample['subject']}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Target New: {sample['target_new']}")
            print(f"Target True: {sample['target_true']}")
            print(f"Portability Data条数: {len(sample['portability_data']) if sample['portability_data'] else 0}")
            print(f"Locality Data条数: {len(sample['locality_data']) if sample['locality_data'] else 0}")
            
            if has_portability > 0 and sample['portability_data']:
                print(f"Portability Data示例: {sample['portability_data'][:1]}")
                print(f"Portability Answer: {sample['portability_answer']}")
            if has_locality > 0 and sample['locality_data']:
                print(f"Locality Data示例: {sample['locality_data'][0]['prompt'] if sample['locality_data'] else ''}")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)