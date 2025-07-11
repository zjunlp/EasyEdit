from dataclasses import dataclass
import torch
from .apply_loreft_intervention_hparam import ApplyLoReFTHyperParams
from torch import nn
class LoReFTIntervention():
    """Phi(h) = h + R^T(Wh + b - Rh)"""
    def __init__(self, layer, device,weight_proj,weight_source,bias_source):
        self.layer = layer
        self.device = device
        self.W_proj = weight_proj.to(device)
        self.W_source = weight_source.to(device)
        self.b_source = bias_source.to(device)

    def forward(self, h):
        rotated_base = torch.bmm(h.float(), self.W_proj)
        output = h + torch.bmm(
            ((torch.bmm(h, self.W_source) + self.b_source) - rotated_base), # batch_size, seq_len, low_rank_dimension
           self.W_proj.transpose(-1, -2)
        )
        return output.to(h.dtype)


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
    model, _ = get_model(hparams)
    reft_layers = hparams.reft_layers
    method = "loreft"
    W_projs = {}
    W_sources = {}
    b_sources = {}
    weight_keys = list(weight.keys())
    bias_keys = list(bias.keys())
    for layer in reft_layers:
        for weight_key in weight_keys:
            if weight_key.startswith(f"layer_{layer}"):
                if(weight_key.endswith("proj_weight")):
                    W_projs[layer] =  nn.Parameter(weight[weight_key])
                elif(weight_key.endswith("source_weight")):
                    W_sources[layer] =  nn.Parameter(weight[weight_key])
        for bias_key in bias_keys:
            if bias_key.startswith(f"layer_{layer}"):
                b_sources[layer] =  nn.Parameter(bias[bias_key])
    for layer in reft_layers:
        intervention_cls = LoReFTIntervention(
            layer=layer,
            device=device,
            weight_proj=W_projs[layer],
            weight_source=W_sources[layer],
            bias_source=b_sources[layer]
        )
        activations = {
            "intervention_cls": intervention_cls,
        }
        model.set_add_activations(layer=layer, activations=activations,method_name=method)
    return model
    