import os
import torch
from ...vector_generators.lm_steer import Hack_no_grad
from .apply_reps_vector_hparam import ApplyRepsHyperParams
         
def reset_reps_layers(model, layers):
    """Reset only the REPS activations for specified layers"""
    model = model.model
    for layer in layers:
        if hasattr(model, 'model') and (hasattr(model.model, 'layers') or (hasattr(model.model, 'module') and hasattr(model.model.module, 'layers'))):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="reps")
            else:
                model.model.layers[layer].reset(method_name="reps")
        elif hasattr(model,'transformer') and hasattr(model.transformer, 'h') or (hasattr(model.transformer, 'module') and hasattr(model.transformer.module, 'h')):  # for GPT models
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="reps")
            else:
                model.transformer.h[layer].reset(method_name="reps")
        else:
            raise NotImplementedError("Failed to reset REPS activations")

def apply_reps(hparams: ApplyRepsHyperParams, pipline=None, vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        model, _ = get_model(hparams)
    else:
        model = pipline
    print('Apply REPS to model: {}'.format(hparams.model_name_or_path))
    # Reset only REPS activations for specified layers
    reset_reps_layers(model, hparams.layers)
    
    layers = hparams.layers
    multipliers = hparams.multipliers
    concept = hparams.concept
    
    for layer, multiplier in zip(layers, multipliers):
        print(f"Layer {layer}, Concept: {concept}")

        if vector is not None:
            steering_vector = vector[f'concept_{concept}'].to(device)
            print(f"Steering vector: User input vector for layer_{layer}")
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir,'concept_'+concept, f"concept_{concept}.pt"
            )
            steering_vector = torch.load(vector_path,map_location=device)
            print("Steering vector path: ",vector_path)
        # print(f"Multiplier {multiplier}")

        model.set_add_activations(
            layer, multiplier * steering_vector, method_name="reps"
        )
    return model
