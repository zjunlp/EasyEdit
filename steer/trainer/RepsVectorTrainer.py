
from .PreferenceModelTrainer import PreferenceModelTrainer
import torch
from tqdm.auto import tqdm

class RepsVectorTrainer(PreferenceModelTrainer):

    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceVector'

    def make_model(self, **kwargs):
        """
        create a model with intervention
        """
        from ..models.interventions import RePSVectorIntervention
        
        print("**Getting embed dim from the following model config**")
        
        intervention_type = kwargs.get("intervention_type", "addition") # addition
        if intervention_type == "addition":
            # create a preference vector intervention object
            steer_vector = RePSVectorIntervention(
                embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), # set the embedding dimension
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # set the low rank dimension, 4
                dropout=kwargs.get("dropout", 0.0),                                # set the dropout rate
                intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # set the dropout rate of the intervention positions
            )
        else:
            raise ValueError(f"Intervention type {intervention_type} not supported")

        self.intervention_type = intervention_type
        self.model.steer_vector = steer_vector.to(self.model.device, dtype=self.model.torch_dtype)
        self.model.steer_vector.train()
        
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
        
        for layer in self.layers:
            intervention_copy = self.model.steer_vector  # all layers share the same intervention instance
            self.model.set_intervention(layer, intervention_copy, "reps")
    
 