import os
import json
import torch
from tqdm import tqdm

from .apply_sta_hparam import ApplySTAHyperParams

def reset_sta_layers(model, layers):
    decoder_layers = model._decoder_layers()
    for layer in layers:
        decoder_layers[layer].reset(method_name="sta")

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

        from ...models.interventions import ActivationAddition

        intervention = ActivationAddition(
            steering_vector=steering_vector,
            multiplier=multiplier,
        )
        intervention = intervention.to(device)
        model.set_intervention(layer, intervention, "sta")
    return model
