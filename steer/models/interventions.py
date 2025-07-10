from dataclasses import dataclass
from typing import Optional
from abc import abstractmethod
import torch
from transformers.utils import ModelOutput

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

<<<<<<< HEAD
class ActivationAddition(BaseIntervention):
    """
    Unified activation addition intervention for steering vectors.
    Supports CAA, VectorPrompt, SAEFeature, and STA methods.
=======
class CAAIntervention(BaseIntervention):
    """
    对比激活添加（CAA）干预类
    直接在激活上添加预计算的引导向量
    """
    def __init__(self, steering_vector, multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.multiplier = multiplier
    
    def forward(self, base, **kwargs):
        """
        Args:
            base: 基础隐藏状态张量，形状为 (batch_size, seq_len, hidden_dim)
        Returns:
            torch.Tensor: 添加引导向量后的隐藏状态
        """
        # 将引导向量广播到所有位置
        steering_addition = self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        return base + steering_addition
    
    def to(self, device):
        super().to(device)
        if isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = self.steering_vector.to(device)
        return self

class VectorPromptIntervention(BaseIntervention):
    """
    向量提示干预类
    """
    def __init__(self, steering_vector, multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.steering_vector = steering_vector
        self.multiplier = multiplier
    
    def forward(self, base, **kwargs):
        """
        Args:
            base: 基础隐藏状态张量，形状为 (batch_size, seq_len, hidden_dim)
        Returns:
            torch.Tensor: 添加向量提示后的隐藏状态
        """
        steering_addition = self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)
        return base + steering_addition
    
    def to(self, device):
        super().to(device)
        if isinstance(self.steering_vector, torch.Tensor):
            self.steering_vector = self.steering_vector.to(device)
        return self

class SAEFeatureIntervention(BaseIntervention):
    """
    SAE特征干预类
>>>>>>> 2a46a1c5 (Forward Still Bug)
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

<<<<<<< HEAD
=======
class STAIntervention(BaseIntervention):
    """
    STA干预类
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
>>>>>>> 2a46a1c5 (Forward Still Bug)

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
<<<<<<< HEAD
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

        if self.subspaces and "steering_factor" in self.subspaces:
            steering_factor = self.subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            # create the zero and non-zero masks, for different training modes
            zero_mask = steering_factor == 0.0 # bs, 1, 1
            nonzero_mask = steering_factor != 0.0 # bs, 1, 1
            # calculate the null-out steering factor: h - (h@v)/||v||^2 * v, the steering coefficient is (h@v)/||v||^2,bs, s, 1 * bs, 1, 1 = bs, s, 1
            null_it_out_steering_factor = -(latent.unsqueeze(dim=-1) / v_norm**2)*zero_mask  
            # combine the steering factors, bs, s, 1 + bs, s, 1 = bs, s, 1
            combined_steering_factor = null_it_out_steering_factor + (steering_factor + self.proj.bias*nonzero_mask) 
            # apply the dropout based on the positions
            dropout_mask = torch.rand_like(combined_steering_factor.float()) > self.intervention_positions_dropout
            combined_steering_factor *= dropout_mask
            steering_vec = steering_vec * combined_steering_factor # bs, s, d

        modified_activations = activations_to_intervene + steering_vec
=======
        # build the preference vector according to the subspaces
        if self.subspaces and "subspaces" in self.subspaces:
            # if the subspaces are specified, select the corresponding weights for each subspace
            for subspace in self.subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            # if the subspaces are not specified, use the first weight for each batch sample
            for i in range(activations_to_intervene.shape[0]):
                v += [self.proj.weight[0]]

        # stack the preference vectors and adjust the dimension: (batch_size, hidden_dim, 1)
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        # calculate the L2 norm of the vector: (batch_size, 1, 1)
        v_norm = torch.norm(v, dim=1, keepdim=True) # bs, 1, 1
        # calculate the latent representation: through batch matrix multiplication and ReLU activation
        latent = torch.relu((torch.bmm(activations_to_intervene, v) + self.proj.bias).squeeze(dim=-1)) # bs, s, 1
        # transpose the preference vector as the steering vector: (batch_size, 1, hidden_dim)
        steering_vec = v.permute(0, 2, 1) # bs, 1, h
        # apply dropout
        steering_vec = self.dropout(steering_vec)

        if self.subspaces and "steering_factor" in self.subspaces:
            # get the steering factor and adjust the dimension: (batch_size, 1, 1)
            steering_factor = self.subspaces["steering_factor"].unsqueeze(dim=-1).unsqueeze(dim=-1) # bs, 1, 1
            # create the zero and non-zero masks, for different training modes
            zero_mask = steering_factor == 0.0 # bs, 1, 1, 用于null-out训练
            nonzero_mask = steering_factor != 0.0 # bs, 1, 1
            # calculate the null-out steering factor: h - (h@v)/||v||^2 * v, the steering coefficient is (h@v)/||v||^2
            null_it_out_steering_factor = -(latent.unsqueeze(dim=-1) / v_norm**2)*zero_mask # bs, s, 1 * bs, 1, 1 = bs, s, 1
            # combine the steering factors
            combined_steering_factor = null_it_out_steering_factor + (steering_factor + self.proj.bias*nonzero_mask) # bs, s, 1
            # apply the dropout based on the positions
            dropout_mask = torch.rand_like(combined_steering_factor.float()) > self.intervention_positions_dropout
            combined_steering_factor *= dropout_mask
            # apply the combined steering factor to the steering vector
            steering_vec = steering_vec * combined_steering_factor # bs, s, d

        # add the steering vector to the selected activations
        modified_activations = activations_to_intervene + steering_vec

        # if we have sliced the activations, scatter the modified activations back to the original tensor
>>>>>>> 2a46a1c5 (Forward Still Bug)
        if self.intervention_locations is not None:
            output = base.scatter(1, expanded_indices, modified_activations)
        else:
            output = modified_activations

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent]
        )
<<<<<<< HEAD
=======
    
    def to(self, device):
        super().to(device)
        self.proj = self.proj.to(device)
        self.dropout = self.dropout.to(device)
        return self
>>>>>>> 2a46a1c5 (Forward Still Bug)
        
@dataclass
class InterventionOutput(ModelOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """
    output: Optional[Any] = None
    latent: Optional[Any] = None