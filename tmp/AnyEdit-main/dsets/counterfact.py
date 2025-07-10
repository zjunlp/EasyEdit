import json
from pathlib import Path
from transformers import AutoTokenizer

from util.globals import *

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
class CounterFactDataset:

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"AKEW"/"CounterFact.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        data = []
        for i, record in enumerate(raw):
            if model_name == 'Llama3-8B-Instruct':
                data.append(
                    {
                        "id": i,
                        "question": get_llama_without_answer(record["requested_rewrite"]["prompt_full"]),
                        "para_question": get_llama_without_answer(record["paraphrase_prompts"][0]),
                        "answer": record["requested_rewrite"]["fact_new_uns"]+'<|eot_id|>',
                        "sub_question": get_list_llama_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                        "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
                    }
                )
            elif model_name == 'Qwen2.5-7B-Instruct':
                data.append(
                    {
                        "id": i,
                        "question": get_qwen_without_answer(record["requested_rewrite"]["prompt_full"]),
                        "para_question": get_qwen_without_answer(record["paraphrase_prompts"][0]),
                        "answer": record["requested_rewrite"]["fact_new_uns"]+'<|im_end|>',
                        "sub_question": get_list_qwen_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                        "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
                    }
                )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
