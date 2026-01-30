
from .SFTModelTrainer import SFTModelTrainer
import torch
from tqdm.auto import tqdm

class SFTTrainer(SFTModelTrainer):

    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'SFTTrainer'

    def make_model(self, **kwargs):
        """
        create a model with intervention
        """
        
        from ..models.interventions import LocalWeightIntervention
        from ..models.interventions import LoraIntervention
        from ..models.interventions import VectorIntervention
        
        intervention_method = kwargs.get("intervention_method", "vector")
        intervention_type = kwargs.get("intervention_type", "addition") # addition
        if intervention_type == "addition":
            if intervention_method == "vector":
            # create a preference vector intervention object
                if kwargs.get("init_vector", None) is not None and isinstance(kwargs.get("init_vector"), VectorIntervention):
                    print("Using provided init vector intervention")
                    steer_vector = kwargs.get("init_vector")
                else:
                    steer_vector = VectorIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), # set the embedding dimension
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # set the low rank dimension, 4
                    dropout=kwargs.get("dropout", 0.0),
                    intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # set the dropout rate of the intervention positions
                    )
            elif intervention_method == "local_weight":
                if kwargs.get("init_vector", None) is not None and isinstance(kwargs.get("init_vector"), LocalWeightIntervention):
                    print("Using provided init localweight intervention")
                    steer_vector = kwargs.get("init_vector")
                else:
                    steer_vector = LocalWeightIntervention(
                        input_dim=kwargs.get("input_dim", self.model.model.config.hidden_size), # set the input dimension
                        embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), # set the embedding dimension
                        low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # set the low rank dimension, 4
                        dropout=kwargs.get("dropout", 0.0),
                        intervention_components=kwargs.get("intervention_components", "mlp"),                                # set the dropout rate
                        intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # set the dropout rate of the intervention positions
                    )
            elif intervention_method == "lora":
                if kwargs.get("init_vector", None) is not None and isinstance(kwargs.get("init_vector"), LoraIntervention):
                    print("Using provided init lora intervention")
                    steer_vector = kwargs.get("init_vector")
                else:
                    steer_vector = LoraIntervention(
                        input_dim=kwargs.get("input_dim", self.model.model.config.hidden_size), # set the input dimension
                        embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), # set the embedding dimension
                        low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # set the low rank dimension, 4
                        dropout=kwargs.get("dropout", 0.0),
                        intervention_components=kwargs.get("intervention_components", "mlp"),                                # set the dropout rate
                        intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # set the dropout rate of the intervention positions
                    )
            else:
                raise NotImplementedError(f"Intervention method {intervention_method} not implemented")
        else:
            raise ValueError(f"Intervention type {intervention_type} not supported")

        self.intervention_type = intervention_type
        self.model.steer_vector = steer_vector.to(self.model.device, dtype=self.model.torch_dtype)
        self.model.steer_vector.train()
        
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
        
        # set the model to eval mode and freeze the parameters
        self.model.model.eval()
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        for layer in self.layers:
            intervention_copy = self.model.steer_vector  # all layers share the same intervention instance
            self.model.set_intervention(layer, intervention_copy, self.hparams.alg_name)