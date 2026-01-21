from dataclasses import dataclass
from typing import Optional
from abc import abstractmethod
import torch
from transformers.utils import ModelOutput
import math
from typing import Dict, Optional, Sequence, Union, List, Any

class BaseIntervention(torch.nn.Module):
    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainable = False
        self.is_source_constant = False

        self.keep_last_dim = kwargs.get("keep_last_dim", False)
        self.use_fast = kwargs.get("use_fast", False)
        self.subspace_partition = kwargs.get("subspace_partition", None)
        # we turn the partition into list indices
        if self.subspace_partition is not None:
            expanded_subspace_partition = []
            for subspace in self.subspace_partition:
                if len(subspace) == 2 and isinstance(subspace[0], int):
                    expanded_subspace_partition.append([i for i in range(subspace[0],subspace[1])])
                else:
                    # it could be discrete indices.
                    expanded_subspace_partition.append(subspace)
            self.subspace_partition = expanded_subspace_partition
            
        if kwargs.get("embed_dim") is not None:
            self.register_buffer('embed_dim', torch.tensor(kwargs["embed_dim"]))
            self.register_buffer('interchange_dim', torch.tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None
            
        if kwargs.get("source_representation") is not None:
            self.is_source_constant = True
            self.register_buffer('source_representation', kwargs["source_representation"])
        else:
            if kwargs.get("hidden_source_representation") is not None:
                self.is_source_constant = True
            else:
                self.source_representation = None
                
    def set_source_representation(self, source_representation):
        self.is_source_constant = True
        self.register_buffer('source_representation', source_representation)
                
    def set_interchange_dim(self, interchange_dim):
        if not isinstance(interchange_dim, torch.Tensor):
            # Convert integer or list into torch.Tensor.
            self.interchange_dim = torch.tensor(interchange_dim)
        else:
            self.interchange_dim = interchange_dim
            
    @abstractmethod
    def forward(self, base, source, subspaces=None):
        pass

class ActivationAddition(BaseIntervention):
    """
    Unified activation addition intervention for steering vectors.
    Supports CAA, VectorPrompt, SAEFeature, and STA methods.

    """
    def __init__(self, steering_vector, multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.multiplier = multiplier
    
    def forward(self, base, **kwargs):
        steering_addition = self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)
        return base + steering_addition
    
    def to(self, device):
        super().to(device)
        if isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = self.steering_vector.to(device)
        return self

class RePSVectorIntervention(BaseIntervention, torch.nn.Module):
    """
    Preference Vector Intervention
    """
    def __init__(self, **kwargs):
       
        super().__init__(**kwargs, keep_last_dim=True)
        
        # create a linear projection layer, map the embedding dimension to the low rank dimension
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        # set the dropout layer
        dropout = kwargs.get("dropout", 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        # set the dropout probability for the intervention positions
        self.intervention_positions_dropout = kwargs.get("intervention_positions_dropout", 0.0)
        self.subspaces = None
        self.intervention_locations = None
        # initialize the bias of the projection layer to 0
        with torch.no_grad():
            self.proj.bias.fill_(0)
            if kwargs.get('init_vector', None) is not None:
                self.proj.weight.copy_(kwargs['init_vector'].to(self.proj.weight.dtype))
                print("Set steering vector no grad!")
                self.proj.weight.requires_grad_(False)
                self.proj.bias.requires_grad_(False)

    def forward(self, base, **kwargs):
       
        # print(f"[DEBUG] base.shape: {base.shape}")
        # print(f"[DEBUG] base[0:2, 4:10, 0:10]:\n{base[0:2, 4:10, 0:10]}")
        
        # Determine the activations to intervene on
        if self.intervention_locations is not None:
            # intervention_locations shape: (batch_size, 1, num_locations) -> (batch_size, num_locations)
            indices = self.intervention_locations.squeeze(1).long()
            # expanded_indices for gather/scatter: (batch_size, num_locations, hidden_dim)
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, base.shape[-1])
            # Gather the activations at specified locations
            activations_to_intervene = torch.gather(base, 1, expanded_indices)
        else:
            activations_to_intervene = base
            expanded_indices = None
         
        v = []
        # for each batch sample, select the corresponding weight for the subspace
        if self.subspaces and "subspaces" in self.subspaces:
            for subspace in self.subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(activations_to_intervene.shape[0]):
                v += [self.proj.weight[0]]

        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        # calculate the L2 norm of the vector: (batch_size, 1, 1)
        v_norm = torch.norm(v, dim=1, keepdim=True)
        # prepare for the null-out training
        latent = torch.relu((torch.bmm(activations_to_intervene, v) + self.proj.bias).squeeze(dim=-1)) # bs, s, 1
        steering_vec = v.permute(0, 2, 1) # bs, 1, h
        steering_vec = self.dropout(steering_vec)

        if self.subspaces and self.subspaces[0]["steering_factor"] is not None:
            steering_factor = self.subspaces[0]["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            # create the zero and non-zero masks, for different training modes
            zero_mask = steering_factor == 0.0 # bs, 1, 1
            nonzero_mask = steering_factor != 0.0 # bs, 1, 1
            # calculate the null-out steering factor: h - (h@v)/||v||^2 * v, the steering coefficient is (h@v)/||v||^2,bs, s, 1 * bs, 1, 1 = bs, s, 1
            # import pdb; pdb.set_trace()
            null_it_out_steering_factor = -(latent.unsqueeze(dim=-1) / v_norm**2)*zero_mask  
            # combine the steering factors, bs, s, 1 + bs, s, 1 = bs, s, 1
            combined_steering_factor = null_it_out_steering_factor + (steering_factor + self.proj.bias*nonzero_mask) 
            # apply the dropout based on the positions
            dropout_mask = torch.rand_like(combined_steering_factor.float()) > self.intervention_positions_dropout
            combined_steering_factor *= dropout_mask
            steering_vec = steering_vec * combined_steering_factor # bs, s, d

        modified_activations = activations_to_intervene + steering_vec

        if self.intervention_locations is not None:
            modified_activations = modified_activations.to(base.dtype)
            output = base.scatter(1, expanded_indices, modified_activations)
        else:
            output = modified_activations

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )

class VectorIntervention(BaseIntervention, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.intervention_method = "vector"
        # create a linear projection layer, map the embedding dimension to the low rank dimension
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        # set the dropout layer
        dropout = kwargs.get("dropout", 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        # set the dropout probability for the intervention positions
        self.intervention_positions_dropout = kwargs.get("intervention_positions_dropout", 0.0)
        self.subspaces = None
        self.intervention_locations = None
        # initialize the bias of the projection layer to 0
        with torch.no_grad():
            self.proj.bias.fill_(0)
            if kwargs.get('init_vector', None) is not None:
                self.proj.weight.copy_(kwargs['init_vector'].to(self.proj.weight.dtype))
                print("Set steering vector no grad!")
                self.proj.weight.requires_grad_(False)
                self.proj.bias.requires_grad_(False)

    def forward(self, base, **kwargs):
        
        # Determine the activations to intervene on
        if self.intervention_locations is not None:
            # intervention_locations shape: (batch_size, 1, num_locations) -> (batch_size, num_locations)
            indices = self.intervention_locations.squeeze(1).long()
            # expanded_indices for gather/scatter: (batch_size, num_locations, hidden_dim)
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, base.shape[-1])
            # Gather the activations at specified locations
            activations_to_intervene = torch.gather(base, 1, expanded_indices)
        else:
            activations_to_intervene = base
            expanded_indices = None
         
        v = []
        # for each batch sample, select the corresponding weight for the subspace
        if self.subspaces and "subspaces" in self.subspaces:
            for subspace in self.subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(activations_to_intervene.shape[0]):
                v += [self.proj.weight[0]]

        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        # calculate the L2 norm of the vector: (batch_size, 1, 1)
        v_norm = torch.norm(v, dim=1, keepdim=True)
        # prepare for the null-out training
        latent = torch.relu((torch.bmm(activations_to_intervene.float(), v.float()) + self.proj.bias.float()).squeeze(dim=-1)) # bs, s, 1
        latent = latent.to(base.dtype)
        steering_vec = v.permute(0, 2, 1) # bs, 1, h
        steering_vec = self.dropout(steering_vec)

        if self.subspaces and self.subspaces[0]["steering_factor"] is not None:
            steering_factor = self.subspaces[0]["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            # create the zero and non-zero masks, for different training modes
            zero_mask = steering_factor == 0.0 # bs, 1, 1
            nonzero_mask = steering_factor != 0.0 # bs, 1, 1
            # calculate the null-out steering factor: h - (h@v)/||v||^2 * v, the steering coefficient is (h@v)/||v||^2,bs, s, 1 * bs, 1, 1 = bs, s, 1
            # import pdb; pdb.set_trace()
            null_it_out_steering_factor = -(latent.unsqueeze(dim=-1) / v_norm**2)*zero_mask  
            # combine the steering factors, bs, s, 1 + bs, s, 1 = bs, s, 1
            combined_steering_factor = null_it_out_steering_factor + (steering_factor + self.proj.bias*nonzero_mask) 
            # apply the dropout based on the positions
            dropout_mask = torch.rand_like(combined_steering_factor.float()) > self.intervention_positions_dropout
            combined_steering_factor *= dropout_mask
            steering_vec = steering_vec * combined_steering_factor # bs, s, d

        modified_activations = activations_to_intervene + steering_vec

        if self.intervention_locations is not None:
            modified_activations = modified_activations.to(base.dtype)
            output = base.scatter(1, expanded_indices, modified_activations)
        else:
            output = modified_activations

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )

class LoraIntervention(BaseIntervention, torch.nn.Module):
    def __init__(self, **kwargs):
        self.intervention_method = "lora"
        print("Initializing PreferenceLoraIntervention with kwargs:", kwargs)
        super().__init__(**kwargs, keep_last_dim=True)
        self.r = kwargs["low_rank_dimension"]
        self.lora_alpha = kwargs["alpha"] if "alpha" in kwargs else kwargs["low_rank_dimension"]
        self.dropout = kwargs.get("dropout", 0.0)
        self.intervention_components = kwargs.get("intervention_components", "mlp")
        self.subspaces = None
        self.multiplier = kwargs.get("multiplier", None)
        self.ablation_vector = kwargs.get("ablation_vector", None)

        if self.dropout > 0.0:
            self.lora_dropout = torch.nn.Dropout(p=self.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = torch.nn.Parameter(torch.zeros(kwargs["input_dim"], kwargs["low_rank_dimension"]))
        self.lora_B = torch.nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))

        # initialize A the same way as the default for nn.Linear and B to zero
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.lora_A = torch.nn.Parameter(self.lora_A.to(kwargs.get("torch_dtype", torch.float32)))
        self.lora_B = torch.nn.Parameter(self.lora_B.to(kwargs.get("torch_dtype", torch.float32)))

    def forward(
        self, base, source=None, **kwargs
    ):
        original_input = kwargs["args"]
        # Pre-compute B×A for applying steering factor
        lora_weights = self.lora_A @ self.lora_B
        
        if self.subspaces is not None and "steering_factor" in self.subspaces[0]:
            # print("Applying steering factor in PreferenceLoraIntervention")
            self.subspaces = self.subspaces[0]
            # Apply steering factor directly to the weights
            steering_factor = self.subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1).to(base.dtype)  # bs, 1, 1            
            # Apply steering factor directly to the weights
            lora_weights = steering_factor * lora_weights
        elif self.multiplier is not None:
            # print(f"Applying multiplier {self.multiplier} in PreferenceLoraIntervention")
            lora_weights = self.multiplier * lora_weights
    
        lora_output = self.lora_dropout(original_input @ lora_weights)
        # --- 稳健投影与调试代码（替换你原来那段） ---
        if self.ablation_vector is not None:
            # ensure on same device
            device = lora_output.device
            # cast ablation vector to float32 for stable computation
            u = self.ablation_vector.to(device)
            # keep original dtype for final cast if you want
            orig_dtype = lora_output.dtype

            # reshape to (1,1,dim)
            if u.dim() == 1:
                u = u.view(1, 1, -1)

            if u.shape[-1] != lora_output.shape[-1]:
                raise ValueError(f"Dimension mismatch: ablation_vector has dimension {u.shape[-1]}, but lora_output has dimension {lora_output.shape[-1]}")

            # Cast to float32 for stable arithmetic (especially if lora_output is float16)
            u32 = u.to(torch.float32)
            v32 = lora_output.to(torch.float32)

            # diagnostics before projection
            dot_before = torch.sum(v32 * u32, dim=-1)              # shape (B, S)
            max_dot_before = dot_before.abs().max().item()
            u_norm_sq = torch.sum(u32 * u32, dim=-1, keepdim=True) # shape (1,1,1)
            u_norm = torch.sqrt(u_norm_sq)
            u_norm_sq_item = u_norm_sq.view(-1).item()

            print(f"[proj debug] dtype lora_output={orig_dtype}, using float32 for proj")
            print(f"[proj debug] max |dot_before| = {max_dot_before:.6e}")
            print(f"[proj debug] u_norm_sq = {u_norm_sq_item:.6e}, u_norm = {u_norm.view(-1).item():.6e}")

            eps = 1e-8
            if u_norm_sq_item > eps:
                # compute projection in float32
                proj_coeff = (dot_before.unsqueeze(-1) / (u_norm_sq + 1e-12))  # shape (B, S, 1)
                projection32 = proj_coeff * u32                              # shape (B, S, D)
                v32_proj = v32 - projection32

                # diagnostics after
                dot_after = torch.sum(v32_proj * u32, dim=-1)  # shape (B, S)
                max_dot_after = dot_after.abs().max().item()
                proj_norm = projection32.view(projection32.shape[0], -1).norm(dim=-1).max().item()
                v_norm = v32.view(v32.shape[0], -1).norm(dim=-1).max().item()

                print(f"[proj debug] max |dot_after| = {max_dot_after:.6e}")
                print(f"[proj debug] max projection norm = {proj_norm:.6e}, max v norm = {v_norm:.6e}")

                # cast back to original dtype
                lora_output = v32_proj.to(orig_dtype)
            else:
                print("[proj debug] u_norm_sq too small, skipping projection.")
                # keep lora_output unchanged

        output = base + lora_output.to(base.dtype)
        
        return InterventionOutput(
            output=output,
            latent=[None]
        )

class LocalWeightIntervention(BaseIntervention, torch.nn.Module):
    def __init__(self, **kwargs):
        print("Initializing WeightIntervention with kwargs:", kwargs)
        super().__init__(**kwargs, keep_last_dim=True)
        self.intervention_method = "local_weight"
        self.dropout = kwargs.get("dropout", 0.0)
        self.intervention_components = kwargs.get("intervention_components", "mlp_mid")
        self.subspaces = None
        self.multiplier = kwargs.get("multiplier", None)
        self.ablation_vector = kwargs.get("ablation_vector", None)
        if self.dropout > 0.0:
            self.lora_dropout = torch.nn.Dropout(p=self.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.delta_weight = torch.nn.Parameter(torch.zeros(kwargs["input_dim"], self.embed_dim))
        self.delta_bias = torch.nn.Parameter(torch.zeros(self.embed_dim))
        torch.nn.init.zeros_(self.delta_weight)
        torch.nn.init.zeros_(self.delta_bias)
        self.delta_weight = torch.nn.Parameter(self.delta_weight.to(kwargs.get("torch_dtype", torch.float32)))
        self.delta_bias = torch.nn.Parameter(self.delta_bias.to(kwargs.get("torch_dtype", torch.float32)))

    def forward(
        self, base, source=None, **kwargs
    ):
        original_input = kwargs["args"]
        delta_weights = self.delta_weight
        delta_bias = self.delta_bias

        if self.subspaces is not None and "steering_factor" in self.subspaces[0]:
            # print("Applying steering factor in PreferenceLoraIntervention")
            self.subspaces = self.subspaces[0]
            # Apply steering factor directly to the weights
            steering_factor = self.subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1).to(base.dtype)  # bs, 1, 1            
            # Apply steering factor directly to the weights
            delta_weights = steering_factor * delta_weights
            delta_bias = steering_factor * delta_bias.view(1, 1, -1).to(base.dtype)  # bs, 1, D
        elif self.multiplier is not None:
            # print(f"Applying multiplier {self.multiplier} in PreferenceLoraIntervention")
            delta_weights = self.multiplier * delta_weights
            delta_bias = (self.multiplier * delta_bias).view(1, 1, -1).to(base.dtype)
        else:
            delta_bias = delta_bias.view(1, 1, -1).to(base.dtype)

        delta_output = original_input @ delta_weights
        delta_output = delta_output + delta_bias
        delta_output = self.lora_dropout(delta_output)
        # --- 稳健投影与调试代码（替换你原来那段） ---
        if self.ablation_vector is not None:
            # ensure on same device
            device = delta_output.device
            # cast ablation vector to float32 for stable computation
            u = self.ablation_vector.to(device)
            # keep original dtype for final cast if you want
            orig_dtype = delta_output.dtype

            # reshape to (1,1,dim)
            if u.dim() == 1:
                u = u.view(1, 1, -1)

            if u.shape[-1] != delta_output.shape[-1]:
                raise ValueError(f"Dimension mismatch: ablation_vector has dimension {u.shape[-1]}, but lora_output has dimension {lora_output.shape[-1]}")

            # Cast to float32 for stable arithmetic (especially if lora_output is float16)
            u32 = u.to(torch.float32)
            v32 = delta_output.to(torch.float32)

            # diagnostics before projection
            dot_before = torch.sum(v32 * u32, dim=-1)              # shape (B, S)
            max_dot_before = dot_before.abs().max().item()
            u_norm_sq = torch.sum(u32 * u32, dim=-1, keepdim=True) # shape (1,1,1)
            u_norm = torch.sqrt(u_norm_sq)
            u_norm_sq_item = u_norm_sq.view(-1).item()

            print(f"[proj debug] dtype delta_output={orig_dtype}, using float32 for proj")
            print(f"[proj debug] max |dot_before| = {max_dot_before:.6e}")
            print(f"[proj debug] u_norm_sq = {u_norm_sq_item:.6e}, u_norm = {u_norm.view(-1).item():.6e}")

            eps = 1e-8
            if u_norm_sq_item > eps:
                # compute projection in float32
                proj_coeff = (dot_before.unsqueeze(-1) / (u_norm_sq + 1e-12))  # shape (B, S, 1)
                projection32 = proj_coeff * u32                              # shape (B, S, D)
                v32_proj = v32 - projection32

                # diagnostics after
                dot_after = torch.sum(v32_proj * u32, dim=-1)  # shape (B, S)
                max_dot_after = dot_after.abs().max().item()
                proj_norm = projection32.view(projection32.shape[0], -1).norm(dim=-1).max().item()
                v_norm = v32.view(v32.shape[0], -1).norm(dim=-1).max().item()

                print(f"[proj debug] max |dot_after| = {max_dot_after:.6e}")
                print(f"[proj debug] max projection norm = {proj_norm:.6e}, max v norm = {v_norm:.6e}")

                # cast back to original dtype
                lora_output = v32_proj.to(orig_dtype)
            else:
                print("[proj debug] u_norm_sq too small, skipping projection.")
                # keep lora_output unchanged

        output = base + delta_output.to(base.dtype)
        
        return InterventionOutput(
            output=output,
            latent=[None]
        )
@dataclass
class InterventionOutput(ModelOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """
    output: Optional[Any] = None
    latent: Optional[Any] = None