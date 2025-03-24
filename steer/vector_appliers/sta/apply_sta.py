import os
import json
import torch
from tqdm import tqdm

from ...vector_generators.lm_steer import Hack_no_grad

from .apply_sta_hparam import ApplySTAHyperParams
         
def reset_sta_layers(model, layers):
    """Reset only the STA activations for specified layers"""
    model=model.model
    for layer in layers:
        if hasattr(model, 'model') and (hasattr(model.model, 'layers') or (hasattr(model.model, 'module') and hasattr(model.model.module, 'layers'))):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="sta")
            else:
                model.model.layers[layer].reset(method_name="sta")
        elif hasattr(model,'transformer') and hasattr(model.transformer, 'h') or (hasattr(model.transformer, 'module') and hasattr(model.transformer.module, 'h')):  # for GPT models
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="sta")
            else:
                model.transformer.h[layer].reset(method_name="sta")
        else:
            raise NotImplementedError("Failed to reset STA activations")

def apply_sta(hparams: ApplySTAHyperParams,pipline=None,vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        model, _ = get_model(hparams)
    else:
        model = pipline
    print('Apply STA to model: {}'.format(hparams.model_name_or_path))
    # Reset only STA activations for specified layers
    reset_sta_layers(model, hparams.layers)
    
    layers = hparams.layers
    multipliers = hparams.multipliers
    trims = hparams.trims
    for layer, multiplier, trim in zip(layers, multipliers, trims):
        print(f"Layer:{layer}  Mode:{hparams.mode}  Trim:{trim}")
        
        if vector is not None:
            steering_vector = vector[f"layer_{layer}_{hparams.mode}_trim{trim}"].to(device)
            print(f"Steering vector: User input vector for layer_{layer}_{hparams.mode}_trim{trim}")
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir, f"layer_{layer}_{hparams.mode}_trim{trim}.pt"
            )
            steering_vector = torch.load(vector_path, map_location=device)
            print("Steering vector path: ",vector_path)
        print("Steering vector: ",steering_vector)
        print(f"Multiplier {multiplier}")

        model.set_add_activations(
            layer, multiplier * steering_vector, method_name="sta"
        )
    return model
