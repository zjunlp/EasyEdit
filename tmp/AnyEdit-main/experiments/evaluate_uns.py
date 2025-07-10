import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
from pathlib import Path
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from dsets import (
    UnKEDataset,
    CounterFactDataset,
    MQUAKEDataset,
    EditeveryDataset
)

from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from AlphaEdit import AlphaEditHyperParams,apply_AlphaEdit_to_model,get_cov
from AlphaEdit_ARE import AlphaEditAREHyperParams,apply_AlphaEdit_ARE_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from util import nethook
from util.globals import *
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}

DS_DICT = {
    "unke": UnKEDataset,
    "cf": CounterFactDataset,
    "mquake": MQUAKEDataset,
    "editevery": EditeveryDataset,
}
def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    sequential: bool = False,
):
    set_seed()
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    params_path = (HPARAMS_DIR / alg_name / hparams_fname)
    hparams = params_class.from_json(params_path)

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    if hparams.model_name == 'Llama3-8B-Instruct':
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    elif hparams.model_name == 'Qwen2.5-7B-Instruct':
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
    if hparams.model_name == 'Llama3-8B-Instruct':
        tokenizer.pad_token_id = tok.eos_token_id

    if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE"]):
        if not os.path.exists(f"{hparams.model_name}_null_space_project.pt"):
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, f"{hparams.model_name}_null_space_project.pt")
        else:
            P = torch.load(f"{hparams.model_name}_null_space_project.pt")
    batch_size = num_edits
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size else 0)
    edited_data = []

    for batch_index in tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = ds[start_index:end_index]
        random_elements = random.sample(ex_datas, 20)
        # case_result_template = str(run_dir / "{}_edits-case_{}.json")
       
        ex_args = dict(ex_data = random_elements) if any(alg in alg_name for alg in ["unke", "unke_ARE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE"]) else dict()
 
        start = time()
        if any(alg in alg_name for alg in ["unke", "unke_ARE","MEMIT","MEMIT_ARE","AlphaEdit","AlphaEdit_ARE"]):
            weights_copy = apply_algo(model, tok, hparams, batch, **ex_args, **nc_args)
        exec_time = time() - start
        print("Execution took", exec_time)

        start = time()
        if not sequential:
            for data in batch:
                if ds_name in ['unke','cf']:
                    question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
                else:
                    question = tokenizer([data['question']], return_tensors='pt', padding=True)
                #print(question.input_ids) 
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                if batch_index < 10 // batch_size + 1:
                    print(f"question:{data['question']}")
                    print(output[0])
                    if ds_name in ['unke','cf']:
                        print(f"question:{data['para_question']}")
                        print(output[1])
                data['original_prediction'] = output[0]
                if ds_name in ['unke','cf']:
                    data['para_prediction'] = output[1]
                if hparams.model_name == 'Llama3-8B-Instruct':
                    data['answer'] = data['answer'][:-len('<|eot_id|>')]
                elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                    data['answer'] = data['answer'][:-len('<|im_end|>')]
            if ds_name in ['unke','cf','mquake']:
                for data in batch:
                    question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
                    with torch.no_grad():
                        generated_ids = model.generate(
                        input_ids=question['input_ids'].to('cuda'),
                        attention_mask=question['attention_mask'].to('cuda'),
                        do_sample=True,
                        temperature=0.001,# Analysis exp
                        max_new_tokens=512
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                    ]

                    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    # if batch_index < 10 // batch_size + 1:
                    #     print(f"question:{data['sub_question']}")
                    #     print(output)

                    data['sub_pred'] = output
         
            edited_data.extend(batch)
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
    if sequential:
        for data in ds:
            if ds_name in ['unke','cf']:
                question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
            else:
                question = tokenizer([data['question']], return_tensors='pt', padding=True)
            #print(question.input_ids) 
            with torch.no_grad():
                generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if batch_index < 10 // batch_size + 1:
                print(f"question:{data['question']}")
                print(output[0])
                if ds_name in ['unke','cf']:
                    print(f"question:{data['para_question']}")
                    print(output[1])
            data['original_prediction'] = output[0]
            if ds_name in ['unke','cf']:
                data['para_prediction'] = output[1]
            if hparams.model_name == 'Llama3-8B-Instruct':
                data['answer'] = data['answer'][:-len('<|eot_id|>')]
            elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                data['answer'] = data['answer'][:-len('<|im_end|>')]
        if ds_name in ['unke','cf','mquake']:
            for data in ds:
                question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,# Analysis exp
                    max_new_tokens=512
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]

                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # if batch_index < 10 // batch_size + 1:
                #     print(f"question:{data['sub_question']}")
                #     print(output)

                data['sub_pred'] = output
        
        edited_data.extend(ds)
    if sequential:
        path = f'output/{alg_name}_{hparams.model_name}_sequential_{ds_name}_result.json'
    else:
        path = f'output/{alg_name}_{hparams.model_name}_{ds_name}_result.json'
    with open(path, 'w', encoding='utf-8') as json_file: 
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"saving to {path}")

    print("Evaluation took", time() - start)
def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","AlphaEdit_ARE", "MEMIT","MEMIT_ARE", "ROME", "FT", "MEND","unke","unke_ARE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="Llama3-8B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama3-8B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "editevery", "unke","mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        help="sequential editing",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        sequential=args.sequential,
    )
