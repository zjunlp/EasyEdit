import copy as copy_module
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from ...util import nethook
import torch.optim as optim
import logging

import numpy as np
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from .unke_are_hparams import UnkeAREHyperParams

LOG = logging.getLogger(__name__)

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: UnkeAREHyperParams,
    layer: int,
    idxs_dict: dict,
):
    input_ids = tok(batch_data, padding=True, return_tensors="pt").to(f"cuda:{hparams.device}")
    zs_out_dict = {}

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                zs_out = tr.output
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    for k, idxs in idxs_dict.items():
        zs_out_list = []
        for idx in idxs:
            zs_out_list.append(zs_out[k, idx])
        zs_out_dict[k] = zs_out_list
    return zs_out_dict

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters

def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    # 获取输入序列的长度
    batch_size, seq_length = attention_mask.shape
    device = input_tensor.device
    dtype = input_tensor.dtype

    # 创建position_ids
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids[:, past_key_values_length:]

    # 创建因果attention mask
    if past_key_values_length == 0:
        past_key_values_length = 0
    
    # 使用attention_mask创建4D因果掩码
    attention_mask_2d = attention_mask
    attention_mask_4d = _prepare_4d_causal_attention_mask(
        attention_mask_2d,
        (batch_size, seq_length),
        input_tensor,
        past_key_values_length,
    )
    
    return attention_mask_4d, position_ids

def get_causal_mask(input_tensor, attention_mask):
    # 获取输入序列的长度
    batch_size, seq_length = attention_mask.shape
    device = input_tensor.device
    dtype = input_tensor.dtype

    # 创建position_ids
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # 创建因果attention mask
    # 使用attention_mask创建4D因果掩码
    attention_mask_2d = attention_mask
    attention_mask_4d = AttentionMaskConverter._make_causal_mask(
        attention_mask_2d.shape, device=device, dtype=dtype
    )
    
    # Llama3特有的缓存位置信息
    cache_position = torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    return attention_mask_4d, position_ids, cache_position

def apply_unke_are_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests,
    hparams: UnkeAREHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
):
    """
    Apply UNKE-ARE editing to a model for a batch of requests.
    
    Args:
        model: The model to edit
        tok: The tokenizer
        requests: A list of editing requests
        hparams: Hyperparameters for UNKE-ARE
        copy: Whether to copy the model before editing
        return_orig_weights: Whether to return the original weights
        keep_original_weight: Whether to keep the original weights
        
    Returns:
        The edited model and (optionally) a copy of the original weights
    """
    # Make a copy of the model if requested
    if copy:
        model = copy_module.deepcopy(model)
    
    # Verify we're editing a valid model
    assert requests[0]['prompt'] is not None, "Request must include a prompt"
    assert requests[0]['target_new'] is not None, "Request must include a target_new"
    
    # First, collect all necessary data
    batch_data = []
    for request in requests:
        data_instance = {
            "question": request["prompt"],
            "answer": request["target_new"]
        }
        batch_data.append(data_instance)
    
    # Generate example data for training stability
    if hasattr(hparams, 'ex_data_num') and hparams.ex_data_num > 0:
        ex_data = []
        # Generate some random prompts for stability
        # In a real application, these could be pulled from a dataset
        for _ in range(hparams.ex_data_num):
            random_text = tok.decode(torch.randint(1000, 30000, (20,)), skip_special_tokens=True)
            ex_data.append(random_text)
    else:
        # Use default stability examples
        ex_data = [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The Earth revolves around the Sun."
        ]
    
    LOG.info(f"Applying UNKE-ARE to {len(batch_data)} requests")
    
    # Store original parameters for layers we'll be modifying
    preserve_params = []
    for name, params in model.named_parameters():
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()} if return_orig_weights else {}
    
    # Compute z values for each request
    z_layer = hparams.layers[-1]
    zs_dict = {}
    idxs_dict = {}
    for k, data in enumerate(batch_data):
        idxs_list, target_list = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
        )
        idxs_dict[k] = idxs_list
        zs_dict[k] = target_list
    
    batch_question_ans = [
        i['question'] + i['answer'] for i in batch_data
    ]
    
    # Apply edits layer by layer
    for i, layer in enumerate(hparams.layers):
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            f"cuda:{hparams.device}"
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        
        # Compute current z values and targets
        cur_zs_dict = compute_ks(model, tok, batch_question_ans, hparams, z_layer, idxs_dict)
        targets_dict = {}
        for k, cur_zs_list in cur_zs_dict.items():
            zs_list = zs_dict[k]
            targets_list = [(a - b)/(len(hparams.layers) - i) for a, b in zip(zs_list, cur_zs_list)]
            targets_dict[k] = targets_list

        # Get example data for stability
        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            f"cuda:{hparams.device}"
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out
        
        # Setup optimization
        criterion = nn.MSELoss()
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n, m in _layer.named_parameters():
            m.requires_grad = True
            
        params = get_optimizer_params(_layer, hparams.lr)
        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999))
        
        # Add residuals to outputs
        for k, idxs_list in idxs_dict.items():
            for j, idx in enumerate(idxs_list):
                resid = targets_dict[k][j]
                layer_out_ks[k, idx] += resid
        
        # Get appropriate attention masks based on model type
        if 'llama' in hparams.model_name.lower():
            input_causal_mask, input_position_ids, input_cache_position = get_causal_mask(layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids, ex_cache_position = get_causal_mask(stat_in, ex_tok['attention_mask'])
        elif 'qwen' in hparams.model_name.lower():
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in, ex_tok['attention_mask'])
        else:
            # Default case for other models
            input_causal_mask = contexts_tok['attention_mask']
            input_position_ids = None
            ex_causal_mask = ex_tok['attention_mask']
            ex_position_ids = None
            input_cache_position = None
            ex_cache_position = None
        
        # Train the layer
        for step in range(hparams.optim_num_step):
            optimizer.zero_grad()
            
            if 'qwen' in hparams.model_name.lower():
                loss = criterion(_layer(stat_in, attention_mask=ex_causal_mask, position_ids=ex_position_ids)[0], stat_out) + \
                       criterion(_layer(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids)[0], layer_out_ks)
            elif 'llama' in hparams.model_name.lower():
                loss = criterion(_layer(stat_in, attention_mask=ex_causal_mask, position_ids=ex_position_ids, cache_position=ex_cache_position)[0], stat_out) + \
                       criterion(_layer(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids, cache_position=input_cache_position)[0], layer_out_ks)
            else:
                # Default case for other models
                output_ex = _layer(stat_in)
                output_ex = output_ex[0] if isinstance(output_ex, tuple) else output_ex
                
                output_in = _layer(layer_in_ks)
                output_in = output_in[0] if isinstance(output_in, tuple) else output_in
                
                loss = criterion(output_ex, stat_out) + criterion(output_in, layer_out_ks)
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if loss.item() < 1e-4:
                break
                
        LOG.info(f"UNKE-ARE Layer {layer} training completed with final loss: {loss.item():.6f}")
    
    # Return the model and (optionally) the original weights
    return model, weights_copy 