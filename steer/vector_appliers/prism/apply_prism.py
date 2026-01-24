import os
import torch
from ...vector_generators.lm_steer import Hack_no_grad
from .apply_prism_hparam import ApplyPrismHyperParams
         
def reset_prism_layers(model, layers):
    """Reset only the prism activations for specified layers"""
    model = model.model
    for layer in layers:
        if hasattr(model, 'model') and (hasattr(model.model, 'layers') or (hasattr(model.model, 'module') and hasattr(model.model.module, 'layers'))):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="prism")
            else:
                model.model.layers[layer].reset(method_name="prism")
        elif hasattr(model,'transformer') and hasattr(model.transformer, 'h') or (hasattr(model.transformer, 'module') and hasattr(model.transformer.module, 'h')):  # for GPT models
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="prism")
            else:
                model.transformer.h[layer].reset(method_name="prism")
        else:
            raise NotImplementedError("Failed to reset prism activations")

def apply_prism(hparams: ApplyPrismHyperParams, pipline=None, vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        model, _ = get_model(hparams)
    else:
        model = pipline
    print('Apply prism to model: {}'.format(hparams.model_name_or_path))
    # Reset only prism activations for specified layers
    reset_prism_layers(model, hparams.layers)
    
    layers = hparams.layers
    multipliers = hparams.multipliers
    concept_id = hparams.concept_id
    
    for layer, multiplier in zip(layers, multipliers):
        if vector is not None:
            data_states = vector[f'layer_{layer}'].to("cpu")
            print(f"Steering vector: User input vector for layer_{layer}")
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir, f"layer_{layer}.pt"
            )
            data_states = torch.load(vector_path, map_location="cpu")
            print("Steering vector path: ", vector_path)

        from ...models.interventions import VectorIntervention
        from ...models.interventions import LoraIntervention
        from ...models.interventions import LocalWeightIntervention

        if hparams.intervention_method=="vector":
            if concept_id < data_states.shape[0]:
                steering_vector = data_states[concept_id]
            else:
                raise ValueError(f"Concept ID {concept_id} exceeds the number of vectors available: {steering_vector.shape[0]}")
            
            intervention = VectorIntervention(
                multiplier=multiplier,
                embed_dim=model.model.config.hidden_size, # set the embedding dimension
                low_rank_dimension=1,            # set the low rank dimension
                init_vector=steering_vector,
            )

        elif hparams.intervention_method=="lora":
            if isinstance(data_states, dict) and all(isinstance(v, dict) for v in data_states.values()):
            # multi-concept
                if concept_id in data_states:
                    data_state = data_states[concept_id]
                else:
                    raise ValueError(f"Concept ID {concept_id} not found in saved layer_{layer}.pt")
            else:
                # single-concept
                data_state = data_states
 
            lora_A = data_state["lora_A"]
            lora_B = data_state["lora_B"]
            input_dim = lora_A.shape[0]
            embed_dim = lora_B.shape[1]
            intervention = LoraIntervention(
                input_dim=input_dim,
                embed_dim=embed_dim,
                low_rank_dimension=data_state["r"],
                alpha=data_state["alpha"],
                intervention_components=data_state["intervention_components"],
                torch_dtype=lora_A.dtype,
                multiplier=multiplier,
            ) 
            with torch.no_grad():
                intervention.lora_A.copy_(lora_A)
                intervention.lora_B.copy_(lora_B)

        elif hparams.intervention_method=="local_weight":
            if isinstance(data_states, dict) and all(isinstance(v, dict) for v in data_states.values()):
                if concept_id in data_states:
                    data_state = data_states[concept_id]
                else:
                    raise ValueError(f"Concept ID {concept_id} not found in saved layer_{layer}.pt")
            else:
                data_state = data_states
            
            delta_weight = data_state["weight"]
            delta_bias = data_state["bias"]
            input_dim = delta_weight.shape[0]
            embed_dim = delta_weight.shape[1]

            intervention = LocalWeightIntervention(
                input_dim=input_dim,
                embed_dim=embed_dim,
                intervention_components=data_state["intervention_components"],
                torch_dtype=delta_weight.dtype,
                multiplier=multiplier,
            )     
            with torch.no_grad():
                intervention.delta_weight.copy_(delta_weight)
                intervention.delta_bias.copy_(delta_bias)

        # print(f"Steering vector shape: {steering_vector.shape}")
        print(f"Loaded Data State for concept_id {concept_id} (single-concept: {data_states is data_states})")
        
        intervention = intervention.to(device)
        model.set_intervention(layer, intervention, "prism")

    return model
