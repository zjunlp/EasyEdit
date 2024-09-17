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
    # Initialize lists to store the processed data from each batch entry
    len_temp = len(context_templates)
    prompts = [item['prompt'] for item in batch]
    labels = [item['target_new'] for item in batch]
    loc_prompts = [item['loc_prompt'] for item in batch]

    mask_token = -100  # ignore_index of CrossEntropyLoss
    if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
        full_prompt = [tokenizer.apply_chat_template([{"role":"user", "content":templ.format(p)}],
                                        add_generation_prompt=True,
                                        tokenize=False) + ' ' + l
                        for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([tokenizer.apply_chat_template([{"role":"user", "content":templ.format(p)}],
                                    add_generation_prompt=True,
                                    tokenize=False) for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    else:
        full_prompt = [f"{templ.format(p + ' ' + l)}" for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([f"{templ.format(p)}" for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    full_prompt += loc_prompts  # add for subject activation

    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    # Mask the tokens based on hparams.objective_optimization
    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    act_masks = []
    deact_masks = []
    # Iterate through each batch entry and compute act_mask, deact_mask
    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:  # subject: Factual Editing
            subject_token = tokenizer.encode(' ' + loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)
            subject_length = len(subject_token)
            act_mask = torch.zeros_like(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)])
            deact_mask = torch.zeros_like(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)])
            for j, token in enumerate(tokens['input_ids'][int(i*len_temp):int((i+1)*len_temp)]):
                start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
                if start_idx is None:
                    start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token1)
                    subject_length = len(subject_token1)
                act_mask[j][start_idx: start_idx + subject_length] = 1
                deact_mask[j][:start_idx] = 1
                deact_mask[j][start_idx + subject_length:] = 1
        else:  # General Editing
            act_mask = None
            deact_mask = None

        # Append the masks to the lists
        act_masks.append(act_mask)
        deact_masks.append(deact_mask)

    # Convert to tensors and move to the specified device
    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [mask.to(device) if mask is not None else None for mask in deact_masks]

    tokens = {key: val.to(device) for key, val in tokens.items()}
    # tokens:[(bs*(len_temp+1))*sequence_length],actmasks:bs*[len_temp*sequence_length],deact_masks:bs*[len_temp*sequence_length]
    return tokens, act_masks, deact_masks

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

