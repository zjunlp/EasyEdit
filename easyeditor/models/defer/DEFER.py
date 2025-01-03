import torch
from .utils import param_subset, get_logits, parent_module, brackets_to_periods
import transformers
import copy


class DEFER(torch.nn.Module):
    def __init__(self, config, model):
        super(DEFER, self).__init__()
        self.model = model
        layer = config.inner_params[0]
        self.config = config
        self.device = config.device

        # strip weight matrix names from layer names
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
            
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # freeze all old layers
        for n, p in self.model.named_parameters():
            if 'defer' in n or 'predict_values' in n:
                print(n)
                continue
            else:
                p.requires_grad = False
            
        # --- Add adaptors to chosen layers ---
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        if type(original_layer) is not DeferAdaptor:
            setattr(edit_module, layer_name, DeferAdaptor(config, original_layer, transpose=transpose).to(self.device))
            self.original_layer = copy.deepcopy(original_layer)

    def reset_layer(self):
        layer_name = self.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        setattr(edit_module, layer_name, self.original_layer.to(self.device))
        
    def get_inner_layer(self, named_parameters):#, layer_name):
        params = []
        for n, p in named_parameters:
            if "defer" in n or "predict" in n:
                params += list(p)
        return params
    
    def __call__(self, **kwargs):
        # if self.config.experiment.task == "hallucination":
        #     key_id = (kwargs["labels"] == -100).sum() - 1
        #     setattr(eval(f"self.model.{self.layer}"), "key_id", key_id) # Tell GRACE which token to use for its query (default is the last token)
        return self.model(**kwargs)
    
    def get_params(self, named_parameters, names):
        param_dict = dict(named_parameters)
        return [param_dict[n] for n in names]

    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens):
        setattr(eval(f"self.model.{self.layer}"), "untrained", False)
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        optimizer = torch.optim.Adam(self.model.parameters(), config.edit_lr)
        self.losses = []
        
        # Tell the model which token to replace
        # if config["experiment"]["task"] == "hallucination":
        #     key_id = (tokens["labels"] == -100).sum() - 1
        # else:
        key_id = -1
        setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        for i in range(config.n_iter):
            outputs = self.model(**tokens)
            logits, loss = outputs.logits, outputs.loss
            self.losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.loss = loss 
        setattr(eval(f"self.model.{self.layer}"), "training", False)

class DeferAdaptor(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(DeferAdaptor, self).__init__()
        
        self.key_id = -1 # Default to using last token
        self.original_layer = layer
        self.device = layer.weight.device
        self.untrained = True
        
        # GPT has non-transposed weights
        if transpose:
            input_dim = layer.weight.shape[1]
            output_dim = layer.weight.shape[0]
        else:
            input_dim = layer.weight.shape[0]
            output_dim = layer.weight.shape[1]
            
        for n, p in layer.named_parameters():
            p.requires_grad = False
            
        self.defer = torch.nn.Linear(input_dim, 1).to(self.device)
        self.predict_values = torch.nn.Linear(input_dim, output_dim).to(self.device)
        self.threshold = config.threshold
    
    def forward(self, *args):
        layer_out = self.original_layer(*args) # Precompute model's prediction
    
        if self.untrained:
            return layer_out

        token_to_edit = min(self.key_id, args[0].shape[1]-1)

        query = args[0][:, token_to_edit, :] # Pull out query for current instance 

        defer = torch.sigmoid(self.defer(query)) # If over threshold, DEFER
        # self.deferral_val = defer
        values = self.predict_values(query) # Predict new values from small model

        if self.training:
            layer_out = defer*values.unsqueeze(1).repeat_interleave(layer_out.shape[1], 1) + (1-defer)*layer_out
        else:
            layer_out = torch.where((defer >= self.threshold), values, layer_out)
        return layer_out
