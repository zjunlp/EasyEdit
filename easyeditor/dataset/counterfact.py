import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class CounterFactDataset(Dataset):
    """
    Dataset of factual knowledge based on CounterFact.
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
            # 检查是否有requested_rewrite数组
            has_rewrite_array = "requested_rewrite" in record and isinstance(record["requested_rewrite"], list) and len(record["requested_rewrite"]) > 0
            
            # 基本字段 - 处理不同的数据格式
            if has_rewrite_array:
                # MQuAKE-CF 格式 - requested_rewrite 是数组
                rewrite = record["requested_rewrite"][0]
                subject = rewrite.get("subject", "")
                prompt = rewrite.get("prompt", "")
                if subject and prompt and "{}" in prompt:
                    prompt_full = prompt.format(subject)
                else:
                    prompt_full = rewrite.get("prompt_full", "")
                
                target_new_str = rewrite.get("target_new", {}).get("str", "")
                target_true_str = rewrite.get("target_true", {}).get("str", "")
                fact_new_uns = rewrite.get("fact_new_uns", "")
            else:
                # 标准格式或其他格式
                subject = record.get("subject", record.get("requested_rewrite", {}).get("subject", ""))
                prompt_full = record.get("prompt", record.get("requested_rewrite", {}).get("prompt_full", ""))
                target_new_str = record.get("target_new", record.get("requested_rewrite", {}).get("target_new", {}).get("str", ""))
                target_true_str = record.get("ground_truth", record.get("requested_rewrite", {}).get("target_true", {}).get("str", ""))
                fact_new_uns = record.get("descriptive_target", record.get("requested_rewrite", {}).get("fact_new_uns", ""))
            
            # 描述性编辑字段
            descriptive_prompt = f"Please describe the situation regarding {subject} and {target_new_str}"
            
            # 处理可移植性数据
            portability_data = []
            portability_answer = target_new_str  # 默认使用target_new作为答案
            
            # MQuAKE-CF 格式 - 使用questions作为可移植性测试
            if "questions" in record and isinstance(record["questions"], list):
                portability_data = record["questions"]
                if "new_answer" in record:
                    portability_answer = record["new_answer"]
            # 其他格式的可移植性数据
            elif "portability_data" in record:
                portability_data = record["portability_data"]
            elif "paraphrase_prompts" in record:
                portability_data = record["paraphrase_prompts"]
            elif "portability" in record and "Subject_Aliasing" in record["portability"]:
                # 将结构化的可移植性数据转换为列表
                subject_aliasing = record["portability"]["Subject_Aliasing"]
                if subject_aliasing:
                    prompts = [item.get("prompt", "") for item in subject_aliasing if "prompt" in item]
                    portability_data = prompts
            elif has_rewrite_array and "paraphrase_prompts" in record["requested_rewrite"][0]:
                portability_data = record["requested_rewrite"][0]["paraphrase_prompts"]
            elif "requested_rewrite" in record and not has_rewrite_array and "paraphrase_prompts" in record["requested_rewrite"]:
                portability_data = record["requested_rewrite"]["paraphrase_prompts"]
            
            # 处理局部性数据
            locality_data = []
            
            # MQuAKE-CF 格式 - 从requested_rewrite[0]获取局部性数据
            if has_rewrite_array and "unsfact_triplets_GPT" in record["requested_rewrite"][0]:
                locality_data = record["requested_rewrite"][0]["unsfact_triplets_GPT"]
            # 其他格式的局部性数据
            elif "locality_data" in record:
                locality_data = record["locality_data"]
            elif "requested_rewrite" in record and not has_rewrite_array and "unsfact_triplets_GPT" in record["requested_rewrite"]:
                locality_data = record["requested_rewrite"]["unsfact_triplets_GPT"]
            elif "locality" in record and "Relation_Specificity" in record["locality"]:
                locality_data = record["locality"]["Relation_Specificity"]
            
            # 构建数据项
            data_item = {
                "subject": subject,
                "prompt": prompt_full,
                "target_new": target_new_str,
                "ground_truth": target_true_str,
                "descriptive_prompt": descriptive_prompt,
                "descriptive_target": fact_new_uns,
                "portability_data": portability_data,
                "portability_answer": portability_answer,  # 新增字段，存储可移植性问题的答案
                "locality_data": locality_data,
                
                # 添加KnowEdit格式的字段，方便与其他代码兼容
                "portability_r": record.get("portability", {}).get("Reasoning"),
                "portability_s": record.get("portability", {}).get("Subject_Aliasing"),
                "portability_l": record.get("portability", {}).get("Logical_Generalization"),
                "locality_rs": record.get("locality", {}).get("Relation_Specificity"),
                "locality_f": record.get("locality", {}).get("Forgetfulness")
            }
            
            data.append(data_item)

        if size is not None:
            data = data[:size]
        self._data = data

        # 打印统计信息用于调试
        print(f"\n===== 数据集统计 =====")
        print(f"总样本数: {len(data)}")
        
        # 检查样本数据
        has_portability = sum(1 for item in data if item['portability_data'] and len(item['portability_data']) > 0)
        has_locality = sum(1 for item in data if item['locality_data'] and len(item['locality_data']) > 0)
        
        print(f"有可移植性数据的样本数: {has_portability}")
        print(f"有局部性数据的样本数: {has_locality}")
        
        # 打印部分样本示例
        if len(data) > 0:
            print("\n===== 样本示例 =====")
            sample = data[0]
            print(f"Subject: {sample['subject']}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Target New: {sample['target_new']}")
            print(f"可移植性数据条数: {len(sample['portability_data']) if sample['portability_data'] else 0}")
            print(f"局部性数据条数: {len(sample['locality_data']) if sample['locality_data'] else 0}")
            if has_portability > 0:
                print(f"可移植性数据示例: {sample['portability_data'][:1]}")
                print(f"可移植性答案: {sample['portability_answer']}")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)