<<<<<<< HEAD
<<<<<<< HEAD
import os
=======
>>>>>>> 2a46a1c5 (Forward Still Bug)
from torch.utils.data import DataLoader
import torch, einops
from .utils.data_utils import make_data_module
from ..models.model_wrapper import BaseModelWrapper
=======
from dataclasses import dataclass
import torch, einops, os
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from transformers import set_seed
import transformers, datasets
from typing import Dict, Optional, Sequence, Union, List, Any

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)

class BaseModelTrainer(object):
    """Base class for all models."""
    def __init__(self, **kwargs):
        pass

    def __str__(self):
        pass

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass

    def train(self, examples, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass

    def load(self, dump_dir, **kwargs):
        pass

    def get_logits(self, concept_id, k=10):
        pass

    def to(self, device):
        pass

<<<<<<< HEAD
class ModelTrainer(BaseModelTrainer):

    def __init__(self, model:BaseModelWrapper, layers, hparams=None, **kwargs):
        self.model = model
        self.layers = layers
        self.hparams = hparams
        self.num_of_layers = len(self.layers) if self.layers else 1
        self.use_wandb = kwargs.get("use_wandb", False)

=======

class ModelTrainer(BaseModelTrainer):

    def __init__(self, model, layer, hparams=None, **kwargs):
        
        self.model = model
        self.layer = layer
        self.hparams = hparams
        self.max_activations = {}
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)
        self.use_wandb = kwargs.get("use_wandb", False)

    # 添加属性访问器，直接使用model的属性
    @property
    def device(self):
        return self.model.device
    
    @property
    def tokenizer(self):
        return self.model.tokenizer
    
    @property
    def model(self):
        return self.model.model

>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.model.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.hparams.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
<<<<<<< HEAD
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.pt")
        weight = self.model.steer_vector.proj.weight.data.cpu()
=======
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
<<<<<<< HEAD
        bias_file = os.path.join(dump_dir, f"{model_name}_bias.pt")
        bias = self.model.steer_vector.proj.bias.data.cpu()
=======
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        priority_mode = kwargs.get("priority_mode", "compute_priority")
        self.priority_mode = priority_mode
        if priority_mode == "mem_priority":
            # prioritize MEM
            concept_id = kwargs.get("concept_id")
            model_name = kwargs.get("model_name", self.__str__())
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
                mmap=True  # Enable memory mapping
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
                mmap=True  # Enable memory mapping
            )
            weight_rank_1 = weight[concept_id].unsqueeze(0)
            bias_rank_1 = bias[concept_id].unsqueeze(0)
            # load only 1 rank to prevent OOM, and faster inference
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight_rank_1.to(self.device)
            self.ax.proj.bias.data = bias_rank_1.to(self.device)
        elif priority_mode == "compute_priority":
            # prioritize COMPUTE
            model_name = kwargs.get("model_name", self.__str__())
            print(f"Loading {model_name} from {dump_dir}.")
            weight = torch.load(
                f"{dump_dir}/{model_name}_weight.pt",
                map_location=torch.device("cpu"),
            )
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt",
                map_location=torch.device("cpu"),
            )
            # override low_rank_dimension in kwargs
            kwargs["low_rank_dimension"] = weight.shape[0]
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight.to(self.device)
            self.ax.proj.bias.data = bias.to(self.device)
    
<<<<<<< HEAD
=======

>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = W_U * (self.model.model.norm.weight +
                        torch.ones_like(self.model.model.norm.weight))[:, None]
            W_U -= einops.reduce(
                W_U, "d_model d_vocab -> 1 d_vocab", "mean"
            )

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]
            
            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits
 
<<<<<<< HEAD
    def to(self, device):
        """Move model to specified device"""
        # Need to be checked carefully
=======

    def to(self, device):
        """Move model to specified device"""
        # 更新model的设备
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        self.model.device = device
        if hasattr(self, 'ax'):
            self.ax = self.ax.to(device)
            if hasattr(self, 'ax_model'):
<<<<<<< HEAD
                # if isinstance(self.ax_model, IntervenableModel):
                #     self.ax_model.set_device(device)
                # else:
                self.ax_model = self.ax_model.to(device)
=======
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
        return self
