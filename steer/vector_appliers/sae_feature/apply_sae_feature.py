import os
import json
import torch
from tqdm import tqdm

from ...vector_generators.lm_steer import Hack_no_grad

from .apply_sae_feature_hparam import ApplySaeFeatureHyperParams
         
def reset_sae_feature_layers(model, layers):
    """Reset only the SaeFeature activations for specified layers"""
    model=model.model
    for layer in layers:
        if hasattr(model, 'model') and (hasattr(model.model, 'layers') or (hasattr(model.model, 'module') and hasattr(model.model.module, 'layers'))):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="sae_feature")
            else:
                model.model.layers[layer].reset(method_name="sae_feature")
        elif hasattr(model,'transformer') and hasattr(model.transformer, 'h') or (hasattr(model.transformer, 'module') and hasattr(model.transformer.module, 'h')):  # for GPT models
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="sae_feature")
            else:
                model.transformer.h[layer].reset(method_name="sae_feature")
        else:
            raise NotImplementedError("Failed to reset SaeFeature activations")

def apply_sae_feature(hparams: ApplySaeFeatureHyperParams,pipline=None,vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        model, _ = get_model(hparams)
    else:
        model = pipline
    print('Apply SaeFeature to model: {}'.format(hparams.model_name_or_path))
    # Reset only SaeFeature activations for specified layers
    reset_sae_feature_layers(model, hparams.layers)
    
    layers = hparams.layers
    multipliers = hparams.multipliers
    for layer, multiplier in zip(layers, multipliers):
        print(f"Layer:{layer}")

        if vector is not None:
            steering_vector = vector[f'layer_{layer}'].to(device)
            print(f"Steering vector: User input vector for layer_{layer}")
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir, f"layer_{layer}.pt"
            )
            steering_vector = torch.load(vector_path, map_location=device)
            print("Steering vector path: ",vector_path)
        print("Steering vector: ",steering_vector)
        print(f"Multiplier {multiplier}")

        model.set_add_activations(
            layer, multiplier * steering_vector, method_name="sae_feature"
        )
    return model
