import torch
from .utils import parent_module, brackets_to_periods
import transformers
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def euc(query, key):
    # Euclidean distance
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return torch.cdist(key, query, p=2)

def perturb_values(chosen_value, num_pert, device):
    # Create a bunch of noised versions of the value, then create batch, then train value
    chosen_value = chosen_value
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0]*0
    noise.requires_grad = True
    chosen_value = chosen_value + noise
    return chosen_value

class GRACE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(GRACE, self).__init__()
        self.config = config
        self.log_dict = {}
        self.model = model
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device

        # --- ensure proper formatting (GRACE edits ~layers~ not weights matrices) ---        
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add GRACE to chosen layers ---
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        setattr(edit_module, layer_name, GRACEAdapter(config, original_layer, transpose=transpose).to(self.device))
        
    def __call__(self, **kwargs):
        # if self.config.task == "hallucination":
        #     print(kwargs)
        #     key_id = (kwargs["labels"] == -100).sum() - 1
        #     setattr(eval(f"self.model.{self.layer}"), "key_id", key_id) # Tell GRACE which token to use for its query (default is the last token)
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def edit(self, config, tokens):
        key_id = (tokens["labels"] == -100).sum() - 1
        setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        
        # --- pass edit label, training mode, and key_id into GRACE ---
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "edit_label", tokens["labels"])
                
        self.losses = []
        # --- train GRACE value ---
        for i in range(config.n_iter):
            # --- insert iteration into each layer (only initiate keys on iteration 1) ---
            setattr(eval(f"self.model.{self.layer}"), "iter", i)
            
            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model(**tokens)
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.Adam(self.model.parameters(), config.edit_lr)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.losses.append(loss.detach().cpu().numpy())
        
        self.loss = loss # Log final loss

        # --- pull out info we want to log from the GRACE layer ---
        setattr(eval(f"self.model.{self.layer}"), "training", False)
        chosen_key = getattr(eval(f"self.model.{self.layer}"), "chosen_key")
        nkeys = len(getattr(eval(f"self.model.{self.layer}"), "keys"))
            
        self.log_dict["chosen_key"] =  chosen_key
        self.log_dict["nkeys"] = nkeys

class GRACEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(GRACEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.init_epsilon = config.eps
        self.dist_fn = config.dist_fn
        self.replacement = config.replacement
        self.device = layer.weight.device
        self.config = config
        self.num_pert = config.num_pert
        self.key_id = -1
    
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False

    def add_key(self, new_key, new_value):
        keys = torch.vstack([self.keys, new_key.detach()]) # Add new key to list of keys

        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True) # Add new value to list of values

        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        epsilons = torch.vstack([self.epsilons, new_epsilon]) # Add new epsilon to list of epsilons

        key_labels = self.key_labels + [self.edit_label] # Add new key_label to list of key_labels

        return keys, values, epsilons, key_labels

    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        return key, value, epsilon, key_label

    def label_match(self, edit_label, key_label):
        return edit_label.float().mean() == key_label.float().mean()

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5 # Cut nearest epsilon in half
        self.epsilons[-1] = smallest_distance / 2 # Cut new epsilon in half
    
    def forward(self, *args):
        # Run layer forward and save what it would have returned for this instance
        layer_out = self.layer(*args)

        ### If training, we need to modify the codebook
        if (not self.training) & ('keys' not in self.__dict__):
            # If it's not training time and we haven't added any keys yet (this is before doing any editing)
            # print(self.__dict__)
            return layer_out
        else:
            token_to_edit = min(self.key_id, args[0].shape[1]-1)
            query = args[0][:, token_to_edit, :] # Just use activation for last token
            if self.config.val_init == "cold":
                new_value = torch.nn.Parameter(torch.rand(1, self.value_shape, requires_grad=True, device=self.device))
            elif self.config.val_init == "warm":
                new_value = torch.nn.Parameter(layer_out[:, token_to_edit, :].detach(), requires_grad=True)

            if 'keys' not in self.__dict__:
                # If no keys exist, initialize keys, values, epsilons, and key labels
                self.keys, self.values, self.epsilons, self.key_labels = self.init_key_value(query, new_value)
            elif self.iter == 0:
                # Keys exist, so we have decide whether or not to update them (the fact that we've made it to this point means there was an error!)

                # --- search through keys for a match for query ---
                dists = torch.cdist(self.keys, query, p=2).view(-1, len(query))
                smallest_distance, nearest_key = dists.min(0)

                if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                    # If there's no close key, make a new key                    
                    self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, new_value)
                else:
                    # If there is a close key, we need to handle conflicts
                    if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                        self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, new_value)
                        self.split_epsilons_in_half(nearest_key, smallest_distance)
                    else:
                        # If the current label is the SAME as the nearest label, just make the nearest epsilon bigger
                        if smallest_distance > self.epsilons[nearest_key]:
                            if self.config.eps_expand== "coverage":
                                self.epsilons[nearest_key] = smallest_distance # Replace nearest epsilon with dist between old key and new key
                            elif self.config.eps_expand == "moving_average":
                                a = 0.5
                                self.keys[nearest_key] = a*self.keys[nearest_key] + (1-a)*query # Move old key to be halfway between
                                self.epsilons[nearest_key] = smallest_distance
                                # self.epsilons[nearest_key] = smallest_distance + self.init_epsilon
            else:
                # If not iter 0, we don't need to change keys, we just need to learn the value
                pass
        # print(token_to_edit)
        # compute distance from query to all keys and find the closest keys
        dists = torch.cdist(self.keys, query, p=2).view(-1, len(query))
        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        if (self.config.val_train == "adv") and (self.training):
            chosen_value = perturb_values(chosen_value, self.num_pert, self.device)

        if self.replacement == "replace_all":
            layer_out = torch.where((smallest_dist <= eps).view(-1, 1, 1), chosen_value.unsqueeze(1).repeat_interleave(layer_out.shape[1], 1), layer_out)
        elif self.replacement == "replace_last":
            layer_out[:, token_to_edit] = torch.where((smallest_dist <= eps), chosen_value, layer_out[:, token_to_edit])
        elif self.replacement == "replace_prompt":
            layer_out[:, :token_to_edit] = torch.where((smallest_dist <= eps), chosen_value, layer_out[:, :token_to_edit])
        else:
            print("token replacement choice not found")
        return layer_out
