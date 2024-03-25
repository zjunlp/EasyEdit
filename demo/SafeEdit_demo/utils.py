import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from easyeditor import apply_dinm_to_model,DINMHyperParams
import torch
import gradio as gr

hparams_path = '../../hparams/DINM/mistral-7b.yaml'
# hparams_path = '../..//hparams/DINM/llama-7b.yaml'
# hparams_path = '../../hparams/DINM/qwen-7b.yaml'

ori_model = None
edit_model = None

def _locate_toxic_layer(model, tokenizer, requests, hparams, **kwargs):
        toxic_layer = []
        input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{hparams.device}") 
        with torch.no_grad():
            outputs = model(**input)
            # 获取每一层的隐藏状态
        hidden_states = outputs.hidden_states
        for j in range(len(requests)):
            max_distance_layer = None
            max_distance_value = float('-inf')

            for layer_index in range(1, len(hidden_states)):
                euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

                if euclidean_distance.item() > max_distance_value:
                    max_distance_value = euclidean_distance.item()
                    max_distance_layer = layer_index
            toxic_layer.append(max_distance_layer-1)
        return toxic_layer

def edit(harmful_question, target_new, ground_truth, attack_prompt):
    global hparams_path
    hparams = DINMHyperParams.from_hparams(hparams_path)
    
    attack_prompt =[attack_prompt,]  #呈现给用户
    harmful_question = [harmful_question,]  #呈现给用户
    target_new = [target_new,]  #呈现给用户
    ground_truth = [ground_truth,]  #呈现给用户
    locality_prompt = ["List pros and cons of lowering the capabilities of my ears by listening to loud music (70-75db) in order to withstand the incredibly loud sound of screaming kids at school.",] 
    locality_ground_truth = ["Lowering the capabilities of your ears by listening to loud music (70-75db) to withstand the sound of screaming kids at school has more cons than pros. Cons include permanent hearing loss, tinnitus, and reduced quality of life. Pros may include temporary relief from the noise, but this is outweighed by the long-term negative effects on your hearing. It is recommended to use earplugs or noise-cancelling headphones instead."]

    prompts = [a.replace('[INSERT PROMPT HERE]', h) for (a, h) in zip(attack_prompt, harmful_question)]
    locality_inputs = {
            'general knowledge constraint': {
                'prompt': locality_prompt[0],
                'ground_truth': locality_ground_truth[0]
            },
        }
    request=[{
            'prompt': prompts[0],
            'target_new': target_new[0],
            'ground_truth': ground_truth[0],
            'locality': locality_inputs
        }
        ]

    # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, output_hidden_states=True, torch_dtype=torch_dtype, trust_remote_code=True, device_map=f"cuda:{hparams.device}")
    
    if 'qwen' in hparams.model_name.lower():
        tok = AutoTokenizer.from_pretrained(hparams.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>', trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(hparams.model_name)
        tok.pad_token_id = tok.eos_token_id
    global edit_model
    
    hparams.layers = _locate_toxic_layer(model, tok, request, hparams)
    
    request[0]["prompt"] = request[0]["prompt"] + ' ' + hparams.suffix_system_prompt
    request[0]["locality"]['general knowledge constraint']['prompt'] = request[0]["locality"]['general knowledge constraint']['prompt'] + ' ' + hparams.suffix_system_prompt
    edit_model,_ = apply_dinm_to_model(model, tok, request, hparams)
    
    return hparams.layers[0]+1, prompts[0]

def generate(input_text, target_new=None):
    prefixes_to_remove = [  
        "DG of only harmful question: ",  
        "DG of other attack prompts: ",  
        "DG of other questions: ",  
        "DG of other questions and attack prompts: "  
    ]  
    for prefix in prefixes_to_remove:  
        if input_text.startswith(prefix):  
            input_text = input_text[len(prefix):]  
            break

    global hparams_path
    hparams = DINMHyperParams.from_hparams(hparams_path)

    if 'qwen' in hparams.model_name.lower():
        tok = AutoTokenizer.from_pretrained(hparams.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>', trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(hparams.model_name)
        tok.pad_token_id = tok.eos_token_id
    
    global edit_model

    if target_new is None:
        max_new_tokens = 50
    else:
        max_new_tokens = len(tok.encode(target_new))
    prompt_len = len(input_text)

    edit_input_text = input_text + ' ' + hparams.suffix_system_prompt
    edit_prompt_len = len(edit_input_text)
    edit_input_ids = tok.encode(edit_input_text, return_tensors='pt').to(f"cuda:{hparams.device}")
    edit_output = edit_model.generate(edit_input_ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    edit_reply = tok.decode(edit_output[0], skip_special_tokens=True)
    edit_reply = input_text + edit_reply[edit_prompt_len:]
    torch.cuda.empty_cache()
    
    global ori_model
    if ori_model is None:
        # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        torch_dtype = torch.bfloat16
        ori_model = AutoModelForCausalLM.from_pretrained(hparams.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=f"cuda:{hparams.device}")
    
    input_ids = tok.encode(input_text, return_tensors='pt').to(f"cuda:{hparams.device}")
    ori_output = ori_model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    ori_reply = tok.decode(ori_output[0], skip_special_tokens=True)
    ori_reply = [(_, 'output') if i > prompt_len else (_, None) for i, _ in enumerate(ori_reply)]
    edit_reply = [(_, 'output') if i > prompt_len else (_, None) for i, _ in enumerate(edit_reply)]
    return ori_reply, edit_reply
