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
            # 检查数据格式
            if "requested_rewrite" not in record:
                continue
                
            # 基本字段提取
            rewrite = record["requested_rewrite"]
            
            # 提取主体
            subject = rewrite.get("subject", "")
            
            # 提取提示
            prompt = rewrite.get("prompt", "")
            if not prompt or not subject:
                continue
                
            # 构建完整提示
            if "{}" in prompt:
                prompt_full = prompt.format(subject)
            else:
                prompt_full = prompt
                
            # 提取目标答案
            target_new = rewrite.get("target_new", {}).get("str", "")
            target_true = rewrite.get("target_true", {}).get("str", "")
            
            if not target_new or not target_true:
                continue
            
            # 提取迁移性数据
            paraphrase_prompts = record.get("paraphrase_prompts", [])
            generation_prompts = record.get("generation_prompts", [])
            
            # 提取局部性数据
            neighborhood_prompts = record.get("neighborhood_prompts", [])
            
            # 构建数据项
            data_item = {
                "subject": subject,
                "prompt": prompt,
                "prompt_full": prompt_full,
                "target_new": target_new,
                "target_true": target_true,
                "paraphrase_prompts": paraphrase_prompts,
                "generation_prompts": generation_prompts,
                "neighborhood_prompts": neighborhood_prompts
            }
            
            # 如果有coupled_prompts_and_properties字段，也加入数据项
            if "coupled_prompts_and_properties" in record:
                data_item["coupled_prompts_and_properties"] = record["coupled_prompts_and_properties"]
            
            data.append(data_item)

        if size is not None:
            data = data[:size]
        self._data = data

        # 打印统计信息用于调试
        print(f"\n===== 数据集统计 =====")
        print(f"总样本数: {len(data)}")
        
        # 检查样本数据
        has_paraphrase = sum(1 for item in data if item['paraphrase_prompts'] and len(item['paraphrase_prompts']) > 0)
        has_generation = sum(1 for item in data if item['generation_prompts'] and len(item['generation_prompts']) > 0)
        has_neighborhood = sum(1 for item in data if item['neighborhood_prompts'] and len(item['neighborhood_prompts']) > 0)
        
        print(f"有paraphrase_prompts的样本数: {has_paraphrase}")
        print(f"有generation_prompts的样本数: {has_generation}")
        print(f"有neighborhood_prompts的样本数: {has_neighborhood}")
        
        # 打印部分样本示例
        if len(data) > 0:
            print("\n===== 样本示例 =====")
            sample = data[0]
            print(f"Subject: {sample['subject']}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Prompt Full: {sample['prompt_full']}")
            print(f"Target New: {sample['target_new']}")
            print(f"Target True: {sample['target_true']}")
            print(f"Paraphrase Prompts条数: {len(sample['paraphrase_prompts']) if sample['paraphrase_prompts'] else 0}")
            print(f"Generation Prompts条数: {len(sample['generation_prompts']) if sample['generation_prompts'] else 0}")
            print(f"Neighborhood Prompts条数: {len(sample['neighborhood_prompts']) if sample['neighborhood_prompts'] else 0}")
            
            if has_paraphrase > 0 and sample['paraphrase_prompts']:
                print(f"Paraphrase Prompts示例: {sample['paraphrase_prompts'][:1]}")
            if has_neighborhood > 0 and sample['neighborhood_prompts']:
                print(f"Neighborhood Prompts示例: {sample['neighborhood_prompts'][:1]}")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)