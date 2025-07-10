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
    """
    Dataset for LongForm data format which includes coupled entities.
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        
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

        # 加载数据文件
        with open(data_dir, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            # 检查数据格式类型
            is_new_format = "subject" in record and "src" in record
            is_requested_rewrite_format = "requested_rewrite" in record
            
            # 初始化数据项（使用统一的字段结构）
            data_item = {
                "subject": "",
                "prompt": "",
                "target_new": "",
                "target_true": "",
                # 使用与counterfact.py一致的命名：迁移性测试
                "portability_data": [],
                "portability_answer": "",
                # 使用counterfact.py风格的KnowEdit格式字段 - 迁移性
                "portability_s": None,  # Subject_Aliasing
                "portability_r": None,  # Reasoning
                "portability_l": None,  # Logical_Generalization
                # 使用与counterfact.py一致的命名：局部性测试
                "locality_data": [],
                # 使用counterfact.py风格的KnowEdit格式字段 - 局部性
                "locality_rs": None,    # Relation_Specificity
                "locality_f": None      # Forgetfulness
            }
            
            if is_new_format:
                # 新格式数据处理 - 包含subject和src字段
                data_item["subject"] = record.get("subject", "")
                data_item["prompt"] = record.get("src", "")
                data_item["target_new"] = record.get("pred", "")
                data_item["portability_answer"] = record.get("pred", "")  # 默认使用pred作为迁移性答案
                
                # 处理answers字段（可能是列表或字符串）
                if "answers" in record:
                    if isinstance(record["answers"], list):
                        data_item["target_true"] = record["answers"][0] if record["answers"] else ""
                    else:
                        data_item["target_true"] = record["answers"]
                
                # 处理rephrase字段（迁移性测试）- 统一到portability_data
                if "rephrase" in record:
                    if isinstance(record["rephrase"], str):
                        # 如果是字符串，可能包含"- "分隔的多个提示
                        if record["rephrase"].startswith("- "):
                            portability_prompts = record["rephrase"].split("- ")[1:]
                        else:
                            portability_prompts = [record["rephrase"]]
                    elif isinstance(record["rephrase"], list):
                        portability_prompts = record["rephrase"]
                        
                    # 构建portability_s格式 - 兼容KnowEdit格式
                    data_item["portability_s"] = []
                    for prompt in portability_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                    
                    # 也添加到portability_data
                    data_item["portability_data"] = portability_prompts
                
                # 处理loc和loc_ans字段（局部性测试）- 统一到locality_data
                if "loc" in record and record["loc"]:
                    locality_prompt = record["loc"]
                    locality_answer = record.get("loc_ans", data_item["target_true"])
                    
                    # 构建locality_rs格式 - 兼容KnowEdit格式
                    data_item["locality_rs"] = [{
                        "prompt": locality_prompt,
                        "ground_truth": locality_answer
                    }]
                    
                    # 也添加到locality_data
                    data_item["locality_data"] = [{
                        "prompt": locality_prompt,
                        "subject": data_item["subject"],
                        "target": locality_answer
                    }]
                
            elif is_requested_rewrite_format:
                # 原始格式数据处理
                rewrite = record["requested_rewrite"]
                
                # 提取主体
                data_item["subject"] = rewrite.get("subject", "")
                
                # 提取提示
                prompt = rewrite.get("prompt", "")
                if prompt and data_item["subject"] and "{}" in prompt:
                    data_item["prompt"] = prompt.format(data_item["subject"])
                else:
                    data_item["prompt"] = rewrite.get("prompt_full", prompt)
                
                # 提取目标答案
                data_item["target_new"] = rewrite.get("target_new", {}).get("str", "")
                data_item["target_true"] = rewrite.get("target_true", {}).get("str", "")
                data_item["portability_answer"] = data_item["target_new"]  # 默认使用target_new作为迁移性答案
                
                # 提取迁移性数据 - 统一到portability_data和portability_s
                if "paraphrase_prompts" in record and record["paraphrase_prompts"]:
                    paraphrase_prompts = record["paraphrase_prompts"]
                    data_item["portability_data"] = paraphrase_prompts
                    
                    # 构建portability_s格式
                    data_item["portability_s"] = []
                    for prompt in paraphrase_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                
                # 提取局部性数据 - 统一到locality_data和locality_rs
                if "neighborhood_prompts" in record and record["neighborhood_prompts"]:
                    neighborhood_prompts = record["neighborhood_prompts"]
                    
                    # 构建locality_rs格式
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
                
                # 处理coupled_prompts_and_properties字段
                if "coupled_prompts_and_properties" in record:
                    data_item["coupled_prompts_and_properties"] = record["coupled_prompts_and_properties"]
                
            else:
                # 其他格式，尝试从各种可能的字段中提取数据
                data_item["subject"] = record.get("subject", "")
                prompt = record.get("prompt", record.get("text", ""))
                
                if data_item["subject"] and prompt and "{}" in prompt:
                    data_item["prompt"] = prompt.format(data_item["subject"])
                else:
                    data_item["prompt"] = record.get("prompt_full", prompt)
                
                data_item["target_new"] = record.get("target_new", record.get("pred", ""))
                data_item["target_true"] = record.get("target_true", record.get("ground_truth", ""))
                data_item["portability_answer"] = data_item["target_new"]  # 默认使用target_new作为迁移性答案
                
                # 从answers字段提取target_true（如果之前没设置）
                if not data_item["target_true"] and "answers" in record:
                    if isinstance(record["answers"], list) and record["answers"]:
                        data_item["target_true"] = record["answers"][0]
                    elif isinstance(record["answers"], str):
                        data_item["target_true"] = record["answers"]
                
                # 处理迁移性数据 - 统一到portability_data和portability_s
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
                    # 构建portability_s格式
                    data_item["portability_s"] = []
                    for prompt in portability_prompts:
                        data_item["portability_s"].append({
                            "prompt": prompt,
                            "ground_truth": data_item["target_new"]
                        })
                
                # 处理局部性数据 - 统一到locality_data和locality_rs
                locality_prompts = []
                if "neighborhood_prompts" in record and record["neighborhood_prompts"]:
                    locality_prompts = record["neighborhood_prompts"]
                elif "loc" in record and record["loc"]:
                    locality_prompts = [record["loc"]]
                
                if locality_prompts:
                    # 构建locality_rs格式
                    data_item["locality_rs"] = []
                    data_item["locality_data"] = []
                    
                    for i, prompt in enumerate(locality_prompts):
                        answer = ""
                        if "neighborhood_answers" in record and len(record["neighborhood_answers"]) > i:
                            answer = record["neighborhood_answers"][i]
                        elif "loc_ans" in record and i == 0:  # 只对第一个loc使用loc_ans
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
            
            # 处理coupled_prompts_and_properties字段（如果存在）
            if "coupled_prompts_and_properties" in record:
                data_item["coupled_prompts_and_properties"] = record["coupled_prompts_and_properties"]
            
            # 只有在数据项包含必要字段时才添加
            if data_item["subject"] and data_item["prompt"] and data_item["target_new"]:
                data.append(data_item)

        if size is not None:
            data = data[:size]
        self._data = data

        # 打印统计信息用于调试
        print(f"\n===== 数据集统计 =====")
        print(f"总样本数: {len(data)}")
        
        # 检查样本数据
        has_portability = sum(1 for item in data if item['portability_data'] and len(item['portability_data']) > 0)
        has_portability_s = sum(1 for item in data if item['portability_s'] and len(item['portability_s']) > 0)
        has_locality = sum(1 for item in data if item['locality_data'] and len(item['locality_data']) > 0)
        has_locality_rs = sum(1 for item in data if item['locality_rs'] and len(item['locality_rs']) > 0)
        
        print(f"有迁移性数据的样本数: {has_portability}")
        print(f"有局部性数据的样本数: {has_locality}")
        
        # 打印部分样本示例
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