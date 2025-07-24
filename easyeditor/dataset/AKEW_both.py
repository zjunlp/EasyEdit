import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to

def get_llama_with_answer(que,ans):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_llama_without_answer_cot(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a multi-hop explanation for the next question: {que}<|eot_id|>"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_vicuna_without_answer(que):
    return f"""USER: {que} ASSISTANT:"""
def get_list_llama_without_answer(que, cot):
    if cot == False:
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L
def get_list_qwen_without_answer(que, cot):
    if cot == False:
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_qwen_without_answer_cot(line) for line in que]
    return L

class AKEWUnifiedDataset(Dataset):
    """
    Dataset for AKEW (A Knowledge Editing Welfare) evaluation.
    Supports three dataset types from the AKEW paper:
    - wikiupdate: WikiUpdate dataset 
    - counterfact: CounterFact dataset
    - mquake: MQuAKE-CF dataset
    """

    def __init__(self, data_dir: str, dataset_type: str = "wikiupdate", size: typing.Optional[int] = None, config=None, model_name: str = None, use_unstructured_data: bool = False, *args, **kwargs):
        data_dir = Path(data_dir)
        if use_unstructured_data:
            if dataset_type.lower() == "counterfact":
                akew_loc = data_dir / "AKEW" / "CounterFact.json"
            elif dataset_type.lower() == "mquake":
                akew_loc = data_dir / "AKEW" / "MQuAKE-CF.json"
            elif dataset_type.lower() == "wikiupdate":
                akew_loc = data_dir / "AKEW" / "WikiUpdate.json"
            elif dataset_type.lower() == "unke":
                akew_loc = data_dir / "UnKE" / "final_data_v3.json"
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
        else:
            akew_loc = data_dir

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

        with open(akew_loc, "r") as f:
            self.raw_data = json.load(f)
        self.model_name = model_name
        self.use_unstructured_data = use_unstructured_data
        self.size = size
        if self.use_unstructured_data:
            self._data = self._process_unstructured_format()
        else:
            self._data = self._process_structured_format()
            
        if size is not None:
            self._data = self._data[:size]

    def _process_structured_format(self):
        data = []
        for i, record in enumerate(self.raw_data):
            if self.dataset_type.lower() == "counterfact":
                processed = self._process_counterfact_structured(record, i)
            elif self.dataset_type.lower() == "mquake":
                processed = self._process_mquake_structured(record, i)
            elif self.dataset_type.lower() == "unke":
                processed = self._process_unke_structured(record, i)
            elif self.dataset_type.lower() == "wikiupdate":
                processed = self._process_wikiupdate_structured(record, i)
            else:
                continue
            data.append(processed)
        return data

    def _process_unstructured_format(self):
        data = []
        for i, record in enumerate(self.raw_data):
            if self.dataset_type.lower() == "counterfact":
                processed = self._process_counterfact_unstructured(record, i)
            elif self.dataset_type.lower() == "mquake":
                processed = self._process_mquake_unstructured(record, i)
            elif self.dataset_type.lower() == "unke":
                processed = self._process_unke_unstructured(record, i)
            elif self.dataset_type.lower() == "wikiupdate":
                processed = self._process_wikiupdate_unstructured(record, i)
            else:
                continue
            data.append(processed)
        return data

    def _process_wikiupdate_structured(self, record, case_id):
        rewrite = record["requested_rewrite"]
        return {
            "case_id": case_id,
            "subject": rewrite["subject"],
            "prompt": rewrite["prompt"].format(rewrite["subject"]) + "?",
            "target_new": rewrite["answer_new"],
            "ground_truth": rewrite["answer_true"],
            "rephrase": [rewrite["question"]],
            "locality": record["loc"] ,
            "locality_ans": record["loc_ans"], 
            "portability_prompt": None,
            "portability_ground_truth": None,    
        }
    def _process_counterfact_structured(self, record, case_id):
        rewrite = record["requested_rewrite"]
        return {
            "case_id": case_id,
            "subject": rewrite["subject"],
            "prompt": rewrite["prompt_full"],
            "target_new": rewrite["target_new"]["str"],
            "ground_truth": rewrite["target_true"]["str"],
            "rephrase": record["paraphrase_prompts"],
            "locality": record["neighborhood_prompts"],
            "locality_ans": [rewrite["target_true"]["str"]] * len(record["neighborhood_prompts"]),
            "portability_prompt": None,
            "portability_ground_truth": None,    
        }

    def _process_mquake_structured(self, record, case_id):
        rewrite = record["requested_rewrite"][0]
        return {
            "case_id": case_id,
            "subject": rewrite["subject"],
            "prompt": rewrite["prompt"].format(rewrite["subject"]) + "?",
            "target_new": rewrite["target_new"]["str"],
            "ground_truth": record["answer"],
            "rephrase": rewrite["question"],
            "locality": record["loc"],
            "locality_ans": record["loc_ans"],
            "portability_prompt": record["questions"],
            "portability_ground_truth": [record["new_answer"]]*len(record["questions"]),
        }

    def _process_counterfact_unstructured(self, record, case_id):
        if 'Llama3-8B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_llama_without_answer(record["requested_rewrite"]["prompt_full"]),
                "para_question": get_llama_without_answer(record["paraphrase_prompts"][0]),
                "answer": record["requested_rewrite"]["fact_new_uns"] + '<|eot_id|>',
                "sub_question": get_list_llama_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_qwen_without_answer(record["requested_rewrite"]["prompt_full"]),
                "para_question": get_qwen_without_answer(record["paraphrase_prompts"][0]),
                "answer": record["requested_rewrite"]["fact_new_uns"] + '<|im_end|>',
                "sub_question": get_list_qwen_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }
        else:
            return {
                "id": case_id,
                "question": record["requested_rewrite"]["prompt_full"],
                "para_question": record["paraphrase_prompts"][0],
                "answer": record["requested_rewrite"]["fact_new_uns"],
                "sub_question": [q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5],
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }

    def _process_mquake_unstructured(self, record, case_id):
        rewrite = record["requested_rewrite"][0]
        prompt = rewrite["prompt"].format(rewrite["subject"]) + "?"
        if 'Llama3-8B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_llama_without_answer(prompt),
                "para_question": get_llama_without_answer(record["requested_rewrite"][0]["question"]),
                "answer": record["requested_rewrite"][0]["fact_new_uns"] + '<|eot_id|>',
                "sub_question": get_list_llama_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5]
            }
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_qwen_without_answer(prompt),
                "para_question": get_qwen_without_answer(record["requested_rewrite"][0]["question"]),
                "answer": record["requested_rewrite"][0]["fact_new_uns"] + '<|im_end|>',
                "sub_question": get_list_qwen_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5]
            }
        else:
            return {
                "id": case_id,
                "question": prompt,
                "para_question": record["requested_rewrite"][0]["question"],
                "answer": record["requested_rewrite"][0]["fact_new_uns"],
                "sub_question": [q["prompt"].format(q["subject"]) for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5],
                "sub_answer": [q["target"] for q in record["requested_rewrite"][0]["unsfact_triplets_GPT"]][:5]
            }

    def _process_unke_unstructured(self, record, case_id):
        if 'Llama3-8B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_llama_without_answer(record["question"]),
                "para_question": get_llama_without_answer(record.get("para_question", record["question"])),
                "answer": record["answer"] + '<|eot_id|>',
                "sub_question": get_list_llama_without_answer(record.get("sub_question", []), False),
                "sub_answer": record.get("sub_answer", [])
            }
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_qwen_without_answer(record["question"]),
                "para_question": get_qwen_without_answer(record.get("para_question", record["question"])),
                "answer": record["answer"] + '<|im_end|>',
                "sub_question": get_list_qwen_without_answer(record.get("sub_question", []), False),
                "sub_answer": record.get("sub_answer", [])
            }
        else:
            return {
                "id": case_id,
                "question": record["question"],
                "para_question": record.get("para_question", record["question"]),
                "answer": record["answer"],
                "sub_question": record.get("sub_question", []),
                "sub_answer": record.get("sub_answer", [])
            }

    def _process_wikiupdate_unstructured(self, record, case_id):
        rewrite = record["requested_rewrite"]
        prompt = rewrite["prompt"].format(rewrite["subject"]) + "?"
        
        if 'Llama3-8B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_llama_without_answer(prompt),
                "para_question": get_llama_without_answer(record["requested_rewrite"]["question"]),
                "answer": record["requested_rewrite"]["fact_new_uns"] + '<|eot_id|>',
                "sub_question": get_list_llama_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            return {
                "id": case_id,
                "question": get_qwen_without_answer(prompt),
                "para_question": get_qwen_without_answer(record["requested_rewrite"]["question"]),
                "answer": record["requested_rewrite"]["fact_new_uns"] + '<|im_end|>',
                "sub_question": get_list_qwen_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }
        else:
            return {
                "id": case_id,
                "question": prompt,
                "para_question": record["requested_rewrite"]["question"],
                "answer": record["requested_rewrite"]["fact_new_uns"],
                "sub_question": [q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5],
                "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
            }


    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        loc_data = [b["locality_rs"] if b["locality_rs"]!=None else b["locality_f"] for b in batch]
        loc=[l[0]["prompt"] if isinstance(l[0]["prompt"],str) else l[0]["prompt"][0] for l in loc_data]
        loc_ans = [l[0]["ground_truth"][0] if isinstance(l[0]["ground_truth"][0],str) else l[0]["ground_truth"][0][0] for l in loc_data]

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

        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        loc_data = [b["locality_rs"] if b["locality_rs"]!=None else b["locality_f"] for b in batch]
        loc=[l[0]["prompt"] if isinstance(l[0]["prompt"],str) else l[0]["prompt"][0] for l in loc_data]

        loc_ans = [l[0]["ground_truth"] if isinstance(l[0]["ground_truth"][0],str) else l[0]["ground_truth"][0] for l in loc_data]
        loc_ans = [l if isinstance(l,str) else l[0] for l in loc_ans]

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
        batch = {
            "edit_inner": edit_inner,
            "loc": loc,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
    
