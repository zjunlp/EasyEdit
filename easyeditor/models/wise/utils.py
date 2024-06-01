import transformers
import torch
import os
import struct

CONTEXT_TEMPLATES_CACHE = None

def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2)+1):
        if all(a == b for a, b in zip(list1[i:i+len(list2)], list2)):
            return i
    return None

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

def print_trainable_parameters(model, new_weight, mask_ratio):
    original_parameters = 0
    new_weight_param = 0
    for _, param in new_weight.named_parameters():
        new_weight_param += param.numel()
    for _, param in model.named_parameters():
        original_parameters += param.numel()
    print(f"Original Model params: {original_parameters} || New Weight params: {new_weight_param} || trainable%: {100 * new_weight_param * (1-mask_ratio) / original_parameters}")


def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value

def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")
    
def get_params(model):
    return model.state_dict()

def get_shape(p, model): 
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x

def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100 # ignore_index of CrossEntropyLoss

    # input
    full_prompt = [f"{templ.format(p + ' ' + l)}" for p, l in zip(prompt, label) for templ in context_templates]
    full_prompt += [batch['loc_prompt']] # add for subject activation

    prompt_ids = tokenizer([f"{templ.format(p)}" for p in prompt for templ in context_templates], return_tensors="pt", padding=True, truncation=True)["input_ids"]

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    if batch['loc_prompt'] in batch['prompt']: ## subject: Factual Editing
        subject_token = tokenizer.encode(' ' + batch['loc_prompt'], add_special_tokens=False)
        subject_token1 = tokenizer.encode(batch['loc_prompt'], add_special_tokens=False)
        subject_length = len(subject_token)
        act_mask = torch.zeros_like(tokens['input_ids'][:-1])
        deact_mask = torch.zeros_like(tokens['input_ids'][:-1])
        for i, token in enumerate(tokens['input_ids'][:-1]):
            start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
            if start_idx is None:
                start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token1)
                subject_length = len(subject_token1)
            act_mask[i][start_idx: start_idx + subject_length] = 1
            deact_mask[i][:start_idx] = 1
            deact_mask[i][start_idx + subject_length:] = 1

        act_mask = act_mask.to(device)
        deact_mask = deact_mask.to(device)
    else: # General Editing
        act_mask = None
        deact_mask = None

    tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens, act_mask, deact_mask

class EarlyStopMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.pre = 0
        self.val = 1e9
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.pre = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def stop(self, ):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02

class EditingMeanAct:
    """Computes and stores the average and current value"""

    def __init__(self, min_a=1e9):
        self.reset(min_a=min_a)

    def reset(self, min_a=1e9):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.min_a = min_a

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.min_a = min(self.min_a, val)

    def mean_act(self):
        return self.avg
    def min_act(self):
        return self.min_a

def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tok = tok(
            ["I", "You", "Because", 'Yes', 'Q: '],
            padding=True,
            return_tensors="pt"
        ).to(device)
        for length, n_gen in length_params: 

            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tok.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tok.batch_decode(gen_token, skip_special_tokens=True)
        CONTEXT_TEMPLATES_CACHE = ['{}'] + [_ + ' {}' for _ in CONTEXT_TEMPLATES_CACHE]
        # print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

