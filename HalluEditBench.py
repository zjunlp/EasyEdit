import json
import pandas as pd
from pathlib import Path
import ast

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class HalluEditBench(Dataset):
    """
    Enhanced dataset for HalluEditBench style knowledge editing
    Supports comprehensive evaluation across 4 dimensions
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_path = Path(data_dir)
        
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

        print(f"Loading data from: {data_path}")
        if data_path.suffix == '.csv':
            raw_df = pd.read_csv(data_path)
            data = self.process_csv_data(raw_df)
        else:
            # Handle JSON format
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        if size is not None:
            data = data[:size]
        self._data = data
        print(f"Successfully loaded {len(self._data)} records")

    def process_csv_data(self, df):
        """Process HalluEditBench data with enhanced structure for 4-dimension evaluation"""
        data = []
        for _, row in df.iterrows():
            # Construct proper edit prompt based on relation
            subject = row['subject']
            relation = row['relation']
            object_val = row['object']
            
            # Smart prompt construction based on relation type
            if 'industry' in relation.lower():
                edit_prompt = f"What is the industry of {subject}?"
            elif 'headquarters' in relation.lower() or 'location' in relation.lower():
                edit_prompt = f"Where is the headquarters of {subject}?"
            elif 'founded' in relation.lower():
                edit_prompt = f"When was {subject} founded?"
            elif 'founder' in relation.lower():
                edit_prompt = f"Who founded {subject}?"
            elif 'CEO' in relation.lower():
                edit_prompt = f"Who is the CEO of {subject}?"
            else:
                edit_prompt = f"What is the {relation} of {subject}?"
            
            record = {
                "subject": subject,
                "relation": relation,
                "prompt": edit_prompt,  # Use constructed prompt instead of provided question
                "target_new": object_val,  
                "ground_truth": object_val,
                
                # Efficacy test
                "efficacy": {
                    "prompt": edit_prompt,
                    "ground_truth": object_val
                },
                
                # Generalization tests (5 types)
                "generalization": {
                    "rephrase": {
                        "prompt": row.get('paraphrased_question') if pd.notna(row.get('paraphrased_question', None)) else None,
                        "ground_truth": object_val
                    },
                    "yes": {
                        "prompt": row.get('yes_question') if pd.notna(row.get('yes_question', None)) else None,
                        "ground_truth": "Yes"
                    },
                    "no": {
                        "prompt": row.get('no_question') if pd.notna(row.get('no_question', None)) else None,
                        "ground_truth": "No"
                    },
                    "mcq": {
                        "prompt": row.get('multiple_choice_with_letters') if pd.notna(row.get('multiple_choice_with_letters', None)) else None,
                        "ground_truth": row.get('multiple_choice_labels') if pd.notna(row.get('multiple_choice_labels', None)) else None
                    },
                    "reversed": {
                        "prompt": row.get('reversed_relation_question') if pd.notna(row.get('reversed_relation_question', None)) else None,
                        "ground_truth": subject
                    }
                },
                
                # Portability tests (multi-hop reasoning)
                "portability": {
                    "2hop": {
                        "prompt": row.get('question_2hop') if pd.notna(row.get('question_2hop', None)) else None,
                        "ground_truth": row.get('answer_2hop') if pd.notna(row.get('answer_2hop', None)) else None
                    },
                    "3hop": {
                        "prompt": row.get('question_3hop') if pd.notna(row.get('question_3hop', None)) else None,
                        "ground_truth": row.get('answer_3hop') if pd.notna(row.get('answer_3hop', None)) else None
                    },
                    "4hop": {
                        "prompt": row.get('question_4hop') if pd.notna(row.get('question_4hop', None)) else None,
                        "ground_truth": row.get('answer_4hop') if pd.notna(row.get('answer_4hop', None)) else None
                    },
                    "5hop": {
                        "prompt": row.get('question_5hop') if pd.notna(row.get('question_5hop', None)) else None,
                        "ground_truth": row.get('answer_5hop') if pd.notna(row.get('answer_5hop', None)) else None
                    },
                    "6hop": {
                        "prompt": row.get('question_6hop') if pd.notna(row.get('question_6hop', None)) else None,
                        "ground_truth": row.get('answer_6hop') if pd.notna(row.get('answer_6hop', None)) else None
                    }
                },
                
                # Locality test (should not change)
                "locality": {
                    "prompt": row.get('locality_question') if pd.notna(row.get('locality_question', None)) else None,
                    "ground_truth": None  # We test that this doesn't change
                }
            }
            data.append(record)
        return data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        # Maintain original logic here, mainly used for training
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        
        loc_data = [b["locality"] for b in batch]
        loc = [l["prompt"] if l["prompt"] is not None else "" for l in loc_data]
        loc_ans = [l["ground_truth"] if l["ground_truth"] is not None else "" for l in loc_data]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
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
        loc_dict = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans_dict = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc_dict["decoder_attention_mask"] = loc_ans_dict["attention_mask"]
        loc_dict["labels"] = self.get_edit_labels(loc_ans_dict["input_ids"])

        batch_dict = {
            "edit_inner": edit_inner,
            "loc": loc_dict,
            "raw": batch,
        }
        return dict_to(batch_dict, self.config.device)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        
        loc_data = [b["locality"] for b in batch]
        loc = [l["prompt"] if l["prompt"] is not None else "" for l in loc_data]
        loc_ans = [l["ground_truth"] if l["ground_truth"] is not None else "" for l in loc_data]

        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
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
        loc_dict = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans_dict = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc_dict["decoder_attention_mask"] = loc_ans_dict["attention_mask"]
        loc_dict["labels"] = self.get_edit_labels(loc_ans_dict["input_ids"])

        batch_dict = {
            "edit_inner": edit_inner,
            "loc": loc_dict,
            "raw": batch,
        }
        return dict_to(batch_dict, self.config.device)
    
