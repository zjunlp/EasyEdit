import os
import os.path as path
import sys
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append('..')
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams,  
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    GraceHyperParams,
    MENDHyperParams
)
from easyeditor import BaseEditor
from easyeditor import AKEWUnifiedDataset  
from easyeditor.evaluate.evaluate_uns import eval_akew_unstructured
from easyeditor.models.unke import unkeHyperParams, apply_unke_to_model
from easyeditor.models.unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from easyeditor.models.lora import LoRAHyperParams, apply_lora_to_model
from easyeditor.models.lora_uns import LoRA_uns_HyperParams, apply_lora_uns_to_model
from easyeditor.models.ft import FTHyperParams, apply_ft_to_model
from easyeditor.models.ft_uns import FT_uns_HyperParams, apply_ft_uns_to_model

from easyeditor.util import nethook
from easyeditor.util.globals import *

import argparse

STRUCTURED_ALG_DICT = {
    "FT": FTHyperParams,
    "ROME": ROMEHyperParams,
    "LoRA": LoRAHyperParams,
}

UNSTRUCTURED_ALG_DICT = {
    "unke": (unkeHyperParams, apply_unke_to_model),
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "LoRA_uns": (LoRA_uns_HyperParams, apply_lora_uns_to_model),
    "FT_uns": (FT_uns_HyperParams, apply_ft_uns_to_model),
    
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def eval_akew_structured(result_path, dataset_type):
    if path.exists(result_path):
        
        with open(result_path,'r') as file:
            datas=json.load(file)
        
        Edit_Succ_list=[data_akew['post']['rewrite_acc'][0] for data_akew in datas]
        Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
        print('Edit_Succ:',Edit_Succ)
        
        Rephrase_Succ_list=[data_wikiupdate['post']['rephrase_acc'][0] for data_wikiupdate in datas]
        Rephrase_Succ=sum(Rephrase_Succ_list)/len(Rephrase_Succ_list)*100
        print('Rephrase_Succ:',Rephrase_Succ)
        
        Locality_list=[]
        for data_longform in datas:
            case_list=[]
            for key in data_longform['post']['locality'].keys():
                case_list.append(sum(data_longform['post']['locality'][key])/len(data_longform['post']['locality'][key])*100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:',Overall_locality)
        
        if dataset_type == 'mquake':
            Portability_list=[]
            for data_longform in datas:
                case_list=[]
                for key in data_longform['post']['portability'].keys():
                    case_list.append(sum(data_longform['post']['portability'][key])/len(data_longform['post']['portability'][key])*100)
                if len(case_list) != 0:
                    Portability_list.append(np.mean(case_list))
            Overall_portability = np.mean(Portability_list)
            print('Overall_portability:',Overall_portability)


def run_structured_editing(args):
    print("Running structured data editing...")
    
    editing_hparams = STRUCTURED_ALG_DICT[args.editing_method]
    
    datas = AKEWUnifiedDataset(
        args.data_dir, 
        dataset_type=args.data_type,
        size=args.ds_size,
        use_unstructured_data=False  # 使用结构化数据
    )
    
    prompts=[data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['ground_truth'] for data in datas]
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_prompts =[data['portability_prompt'] for data in datas]
    portability_answers = [data['portability_ground_truth'] for data in datas]
    if args.data_type == 'counterfact':
        locality_prompts = [data['locality'] for data in datas]
        locality_answers = [data['locality_ans'] for data in datas]
    else:
        locality_prompts = [[data['locality']] for data in datas]
        locality_answers = [[data['locality_ans']] for data in datas]
    assert len(prompts)==len(portability_prompts)==len(portability_answers)
    assert len(prompts)==len(locality_prompts)==len(locality_answers)

    locality_inputs = {}
    portability_inputs = {}
    locality_inputs = {
        'locality':{
            'prompt': locality_prompts,
            'ground_truth': locality_answers
        }
    }
    
    portability_inputs = {
        'por_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_answers  
        }       
    }

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    if args.data_type == 'mquake':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject = subject,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            # portability_inputs=portability_inputs,
            subject = subject,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit
        )
    
    return metrics, hparams.model_name

def run_unstructured_editing(args):
    print("Running unstructured data editing...")
    
    set_seed()
    params_class, apply_algo = UNSTRUCTURED_ALG_DICT[args.editing_method]
    hparams = params_class.from_json(args.hparams_dir)
    
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token = tok.eos_token
    
    ds = AKEWUnifiedDataset(
        args.data_dir,
        dataset_type=args.data_type,
        model_name=hparams.model_name,
        size=args.ds_size,
        use_unstructured_data=True
    )
     
    with open(Path(args.data_dir)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    if 'Llama3-8B-Instruct' in hparams.model_name:
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, padding_side='left')
    if 'Llama3-8B-Instruct' in hparams.model_name:
        tokenizer.pad_token_id = tok.eos_token_id
    
    batch_size = args.batch_size
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size else 0)
    edited_data = []

    for batch_index in tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = ds[start_index:end_index]
        random_elements = random.sample(ex_datas, 20)
        # case_result_template = str(run_dir / "{}_edits-case_{}.json")
       
        ex_args = dict(ex_data = random_elements) if any(alg in args.editing_method for alg in ["unke", "unke_ARE"]) else dict()
        
        start = time.time()
        if args.editing_method == "LoRA_uns":
            edited_model, weights_copy = apply_algo(model, tok, hparams, batch, **ex_args)
            model = edited_model 
            weights_copy = {}
        elif  args.editing_method == "FT_uns":
            edited_model, deltas = apply_algo(model, tok, hparams, batch, **ex_args)
            weights_copy = {}
            for k, delta in deltas.items():
                weights_copy[k] = (nethook.get_parameter(model, k) - delta).detach().clone()    
        else:
            weights_copy = apply_algo(model, tok, hparams, batch, **ex_args)
        
        exec_time = time.time() - start
        print("Execution took", exec_time)

        start = time.time()
        if not args.sequential_edit: 
            for data in batch:
                if args.data_type in ['unke','counterfact','mquake','wikiupdate']:
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
                    if args.data_type in ['unke','counterfact','mquake','wikiupdate']:
                        print(f"question:{data['para_question']}")
                        print(output[1])
                data['original_prediction'] = output[0]
                if args.data_type in ['unke','counterfact','mquake','wikiupdate']:
                    data['para_prediction'] = output[1]
                if 'Llama3-8B-Instruct' in hparams.model_name:
                    data['answer'] = data['answer'][:-len('<|eot_id|>')]
                elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
                    data['answer'] = data['answer'][:-len('<|im_end|>')]
            if args.data_type in ['unke','counterfact','mquake','wikiupdate']:
                for data in batch:
                    question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
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
                    # if batch_index < 10 // batch_size + 1:
                    #     print(f"question:{data['sub_question']}")
                    #     print(output)
                    data['sub_pred'] = output

            edited_data.extend(batch)
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
    
    if args.sequential_edit:
        for data in ds:
            if args.data_type in ['unke','counterfact','wikiupdate','mquake']:
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
            data['original_prediction'] = output[0]
            if args.data_type in ['unke','counterfact','wikiupdate','mquake']:
                data['para_prediction'] = output[1]
            if 'Llama3-8B-Instruct' in hparams.model_name:
                data['answer'] = data['answer'][:-len('<|eot_id|>')]
            elif 'Qwen2.5-7B-Instruct' in hparams.model_name:
                data['answer'] = data['answer'][:-len('<|im_end|>')]
        if args.data_type in ['unke','counterfact','mquake','wikiupdate']:
            for data in ds:
                question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
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
                # if batch_index < 10 // batch_size + 1:
                #     print(f"question:{data['sub_question']}")
                #     print(output)
                data['sub_pred'] = output
        
        edited_data.extend(ds)

    print("Evaluation took", time.time() - start)
    return edited_data, hparams.model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, choices=['wikiupdate', 'counterfact', 'mquake'])
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--sequential_edit', action="store_true")
    
    parser.add_argument(
        '--device', 
        default=0, 
        type=int,
        help="CUDA device for evaluation"
    )
    # === new for uns ===
    parser.add_argument(
        '--model_path', 
        default='sentence-transformers/all-MiniLM-L6-v2', 
        type=str,
        help="SentenceTransformer model path for BERT Score calculation"
    )
    parser.add_argument(
        '--use_unstructured_data', 
        action="store_true",
        help="Use unstructured data format instead of structured triplets"
    )
    parser.add_argument(
        '--batch_size', 
        default=1, 
        type=int,
        help="Batch size for unstructured data editing"
    )
    
    args = parser.parse_args()

    if args.use_unstructured_data:
        if args.editing_method not in UNSTRUCTURED_ALG_DICT:
            raise ValueError(f"Algorithm {args.editing_method} is not supported for unstructured data")
        print(f"Using unstructured data editing with {args.editing_method}")
        edited_data, model_name = run_unstructured_editing(args)
        
        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir)
        result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{model_name.split("/")[-1]}_{args.data_type}_AKEW_unstructured_results.json')
        json.dump(edited_data, open(result_path, 'w'), indent=4)
        print(f"Results saved to: {result_path}")
        
        eval_akew_unstructured(result_path, args.data_type, args.model_path, args.device)
        
    else:
        if args.editing_method not in STRUCTURED_ALG_DICT:
            raise ValueError(f"Algorithm {args.editing_method} is not supported for structured data")
        print(f"Using structured data editing with {args.editing_method}")
        metrics, model_name = run_structured_editing(args)
        
        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir)
        result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{model_name.split("/")[-1]}_{args.data_type}_AKEW_structured_results.json')
        json.dump(metrics, open(result_path, 'w'), indent=4)
        print(f"Results saved to: {result_path}")
        
        eval_akew_structured(result_path, args.data_type)