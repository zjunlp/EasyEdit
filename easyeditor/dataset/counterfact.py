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
        cf_loc = data_dir

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

        with open(cf_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            # 基本编辑字段
            subject = record["requested_rewrite"]["subject"]
            prompt_full = record["requested_rewrite"]["prompt_full"]
            target_new_str = record["requested_rewrite"]["target_new"]["str"]
            target_true_str = record["requested_rewrite"]["target_true"]["str"]
            
            # 描述性编辑字段
            descriptive_prompt = f"Please describe the situation regarding {subject} and {target_new_str}"
            fact_new_uns = record["requested_rewrite"]["fact_new_uns"]
            
            # 可移植性测试 - 直接使用paraphrase_prompts
            portability_data = record.get("paraphrase_prompts", [])
            
            # 局部性测试 - 从requested_rewrite中获取unsfact_triplets_GPT
            locality_data = record["requested_rewrite"].get("unsfact_triplets_GPT", [])
            
            # 构建数据项
            data_item = {
                "subject": subject,
                "prompt": prompt_full,
                "target_new": target_new_str,
                "ground_truth": target_true_str,
                "descriptive_prompt": descriptive_prompt,
                "descriptive_target": fact_new_uns,
                "portability_data": portability_data,
                "locality_data": locality_data
            }
            
            data.append(data_item)

        if size is not None:
            data = data[:size]
        self._data = data

        # 打印详细数据用于调试
        print("\n===== 详细数据样本检查 =====")
        
        # 检查全局数据统计
        has_portability = sum(1 for item in data if item['portability_data'] and len(item['portability_data']) > 0)
        has_locality = sum(1 for item in data if item['locality_data'] and len(item['locality_data']) > 0)
        
        print(f"总样本数: {len(data)}")
        print(f"有可移植性数据的样本数: {has_portability}")
        print(f"有局部性数据的样本数: {has_locality}")
        
        # 详细检查前10个样本
        for i, item in enumerate(data[:10]):  # 增加到前10个样本
            print(f"\n============ 样本 {i+1} ============")
            print(f"Subject: {item['subject']}")
            print(f"Prompt: {item['prompt']}")
            print(f"Target New: {item['target_new']}")
            
            print("\n------ 可移植性测试数据 (portability_data) ------")
            print(f"类型: {type(item['portability_data'])}")
            if isinstance(item['portability_data'], list):
                print(f"长度: {len(item['portability_data'])}")
                print("完整内容:")
                for j, p in enumerate(item['portability_data']):
                    print(f"  [{j}]: \"{p}\"")
            else:
                print("不是列表类型")
            
            print("\n------ 局部性测试数据 (locality_data) ------")
            print(f"类型: {type(item['locality_data'])}")
            if isinstance(item['locality_data'], list):
                print(f"长度: {len(item['locality_data'])}")
                if len(item['locality_data']) > 0:
                    print("内容示例:")
                    for j, loc in enumerate(item['locality_data']):
                        print(f"  [{j}]:")
                        if isinstance(loc, dict):
                            for key, value in loc.items():
                                print(f"    {key}: {value}")
                            # 特别检查format字符串是否可以正确格式化
                            if 'prompt' in loc and 'subject' in loc:
                                try:
                                    formatted = loc['prompt'].format(loc['subject'])
                                    print(f"    格式化后: {formatted}")
                                except Exception as e:
                                    print(f"    格式化错误: {str(e)}")
                        else:
                            print(f"    不是字典: {type(loc)}")
                else:
                    print("空列表")
            else:
                print("不是列表类型")
        
        print("\n===== 数据处理前检查完成，提前返回 =====")
        return

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)