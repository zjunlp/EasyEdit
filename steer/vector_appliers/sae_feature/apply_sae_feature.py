import os
import json
import torch
from tqdm import tqdm

from .apply_sae_feature_hparam import ApplySaeFeatureHyperParams

def reset_sae_feature_layers(model, layers):
    decoder_layers = model._decoder_layers()
    for layer in layers:
        decoder_layers[layer].reset(method_name="sae_feature")

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

        from ...models.interventions import ActivationAddition

        intervention = ActivationAddition(
            steering_vector=steering_vector,
            multiplier=multiplier,
        )
        intervention = intervention.to(device)
        model.set_intervention(layer, intervention, "sae_feature")
    return model
