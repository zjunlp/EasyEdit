from dataclasses import dataclass
import pyreft
import transformers
import torch
from typing import Sequence, Dict
from .apply_loreft_intervention_hparam import ApplyLoReFTHyperParams
from torch import nn
from pyvene import SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
class ConceptReFTIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + R^T(Wh + b - Rh)
    Ref: https://arxiv.org/pdf/2404.03592

    Note that this intervention is used for concept-based Direft.
    The main difference is that weights are assumed to be trained and saved as 3D tensors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_proj = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.b_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], kwargs["low_rank_dimension"]))

    def encode(
        self, base, source=None, subspaces=None
    ):
        """High-dimensional concept space."""
        proj_weight = self.W_proj[subspaces["input_subspaces"]] # batch_size, embed_dim, low_rank_dimension
        rotated_base = torch.bmm(base, proj_weight) # [batch_size, seq_len, embed_dim] X [batch_size, embed_dim, low_rank_dimension]

        return rotated_base # batch_size, seq_len, low_rank_dimension

    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_weight = self.W_proj[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_weight = self.W_source[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_bias = self.b_source[subspaces["idx"]].unsqueeze(dim=1) # batch_size, 1, low_rank_dimension

        rotated_base = torch.bmm(base.float(), proj_weight) # batch_size, seq_len, low_rank_dimension
        output = base + torch.bmm(
            ((torch.bmm(base, source_weight) + source_bias) - rotated_base), # batch_size, seq_len, low_rank_dimension
            proj_weight.transpose(-1, -2)
        )
        return output.to(base.dtype)

def apply_loreft(hparams: ApplyLoReFTHyperParams,model = None):
    from ...models import get_model
    # make_model
    dump_dir = hparams.steer_vector_load_dir
    model_name = "loreft"
    weight = torch.load(
        f"{dump_dir}/{model_name}_weight.pt",weights_only= True
    )
    bias = torch.load(
        f"{dump_dir}/{model_name}_bias.pt",weights_only= True
    )
    device = hparams.device
    weight_keys = list(weight.keys())
    n_concepts = weight[weight_keys[0]].shape[0]
    low_rank_dimension = weight[weight_keys[0]].shape[-1]
    model, _ = get_model(hparams)
    reft_layers = hparams.reft_layers
    dtype = model.torch_dtype
    intervention_cls = ConceptReFTIntervention
    reft_config = pyreft.ReftConfig(representations=[{
        "layer": l, "component": "block_output",
        "low_rank_dimension": low_rank_dimension,
        "intervention": intervention_cls(n_concepts=n_concepts,embed_dim=model.model.config.hidden_size,
        low_rank_dimension=low_rank_dimension,dtype =dtype)} for l in reft_layers])
    reft_model = pyreft.get_reft_model(model.model, reft_config)
    reft_model.set_device(device)
    for intervention_name, intervention in reft_model.interventions.items():
        intervention.W_proj.data = weight[f"{intervention_name}.proj_weight"]
        intervention.W_source.data = weight[f"{intervention_name}.source_weight"]
        intervention.b_source.data = bias[f"{intervention_name}.bias"]
    for k,v in reft_model.interventions.items():
        v.eval()
    return reft_model,model
@dataclass
class InterventionEvalDataCollator(object):
    """Collate examples for Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DefaultDataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        intervention_locations will be something like [1,10,0,0,0] where all 0s are padding intervention locations.
        """
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])
            _intervention_location_paddings = torch.tensor(
                [[-1 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))] for _ in range(inst["intervention_locations"].shape[0])]) # pointing to the first padding token
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_locations"] = inst["intervention_locations"] + 1 # shift by 1 to point to the first non-padding token, and all paddings will be 0.

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            offset = max_seq_len - non_pad_len
            inst["intervention_locations"] = inst["intervention_locations"] + offset
            inst["input_ids"] = torch.cat((_input_id_paddings, torch.tensor([self.tokenizer.pad_token_id]), inst["input_ids"])).int()
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs
    