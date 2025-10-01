from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .simie import SimIE
from .simie_hparams import SimIEHyperParams
from ...util import nethook
def apply_simie_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SimIEHyperParams,
    copy=False,
    return_orig_weights=True,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    SimIE is a meta-framework that wraps around base editing methods
    (like ROME, MEMIT) and applies incremental updates to maintain
    editing quality in sequential/lifelong editing scenarios.
    """
    
    from ...util.alg_dict import ALG_DICT
    base_apply_func = ALG_DICT[hparams.base_method]
    base_hparams = hparams.base_hparams
    
    if not hasattr(hparams, '_simie_instance') or hparams._simie_instance is None:
        print("\n" + "="*80)
        print("[SimIE] First call detected - Initializing SimIE")
        init_request = requests[-1] if len(requests) > 1 else requests[0]
        
        _, init_weights = base_apply_func(
            model, 
            tok, 
            [init_request], 
            base_hparams,
            copy=False, 
            return_orig_weights=True, 
            keep_original_weight=False,
            **kwargs
        )
        
        simie = SimIE(
            lamHyper=hparams.lamHyper,
            init=hparams.init_model,
            solver=hparams.solver
        )
        
        simie.initialization(
            model_name=hparams.model_name,
            init_weights=init_weights,
            device=hparams.device,
            fast=hparams.fast
        )
        
        model = simie.reset_parameter(model)
        
        hparams._simie_instance = simie
        hparams._simie_call_count = 0
        hparams._simie_total_edits = 0
        
        print(f"[SimIE] Initialization complete!")
        print("="*80 + "\n")
    else:
        simie = hparams._simie_instance
        hparams._simie_call_count += 1
        
    for request in requests:
        if simie.init:
            model = simie.reset_parameter(model)
        keys_cache = simie.cache(model, [request], tok)
        edited_model, weights_copy = base_apply_func(
            model,
            tok,
            [request],
            base_hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
            **kwargs
        )
        edited_model = simie.update(edited_model, keys_cache)
        model = edited_model
        hparams._simie_total_edits += 1
        
    if return_orig_weights:
        return edited_model, simie.init_weights_copy
    else:
        return edited_model, {}