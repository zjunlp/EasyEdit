import torch 
import os
from .apply_lm_steer_hparam import ApplyLmSteerHyperParams

def apply_lm_steer(hparams: ApplyLmSteerHyperParams,pipline=None, vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        if getattr(hparams, 'vllm_enable', False):
            model, VLLM_model = get_model(hparams)
        else:
            model, tokenizer = get_model(hparams)
    else:
        model = pipline
   
    model.reset_lm_steer()
    model.replace_final_layer(hparams)
    model.model.to(device)  
    print(device)
    if vector is not None:
        model.steer.load_state_dict(vector)
    else:
        ckpt_name = os.path.join(hparams.steer_vector_load_dir, 'lm_steer_vector.pt')
        # Load the steer checkpoint
        ckpt = torch.load(ckpt_name, map_location=device)
        model.steer.load_state_dict(ckpt[1])
   
    model.steer.to(device)
    
    #set the steer values
    steer_values = (torch.tensor(hparams.steer_values, dtype=model.torch_dtype, device=device) 
                if hparams.steer_values is not None else None)
    model.steer.set_value(steer_values[None])
    
    if hparams.vllm_enable:
        return model, VLLM_model
    else:
        return model, tokenizer

def apply_multimodal_lm_steer(hparams: ApplyLmSteerHyperParams,pipline=None, vector=None):
    from ...models.Multimodal_get_model import get_model
    device = hparams.device
    if pipline is None:
        if getattr(hparams, 'vllm_enable', False):
            model, VLLM_model = get_model(hparams)
        else:
            model, tokenizer = get_model(hparams)
    else:
        model = pipline
   
    model.reset_lm_steer()
    model.replace_final_layer(hparams)
    model.model.to(device)  
    print(device)
    if vector is not None:
        model.steer.load_state_dict(vector)
    else:
        ckpt_name = os.path.join(hparams.steer_vector_load_dir, 'lm_steer_vector.pt')
        # Load the steer checkpoint
        ckpt = torch.load(ckpt_name, map_location=device)
        model.steer.load_state_dict(ckpt[1])
   
    model.steer.to(device)
    
    #set the steer values
    steer_values = (torch.tensor(hparams.steer_values, dtype=model.torch_dtype, device=device) 
                if hparams.steer_values is not None else None)
    model.steer.set_value(steer_values[None])
    
    if hparams.vllm_enable:
        return model, VLLM_model
    else:
        return model, tokenizer
    



            