import transformers
import torch
import os
import numpy as np
import datetime
import struct
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

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

def tokenize(batch, tokenizer, device, test=False):
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100 # ignore_index of CrossEntropyLoss
    if test or not label:
        tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    else:
        full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
        prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
        tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        for i in range(len(prompt)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    
    tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens


def multimodal_tokenize(batch, processor, device, hparams):
    prompts = [item['prompt'] for item in batch]
    input_images = [item['image'] for item in batch]
    labels = [item['target'] for item in batch]
    file_type = batch[0]['file_type']
    mask_token = -100 # ignore_index of CrossEntropyLoss
    if file_type == "video":
        temp_prompt = [processor.apply_chat_template([
                                {

                                    "role": "user",
                                    "content": [
                                        {"type": "video"},
                                        {"type": "text", "text": p},
                                        ],
                                },
                            ],
                                            add_generation_prompt=True,
                                            tokenize=False) + l
                        for p, l in zip(prompts, labels)] 
    elif file_type in ["image", "single-image", "multi-image"]:
        if file_type == "multi-image":
            num_images = len(input_images[0])
        else:
            num_images = 1
        
        temp_prompt = [processor.apply_chat_template([
                                {

                                    "role": "user",
                                    "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}],
                                },
                            ],
                                            add_generation_prompt=True,
                                            tokenize=False)  + l
                        for p, l in zip(prompts, labels)]
    else:
        raise AssertionError("Not support file type: {}".format(file_type))
    
    full_prompt = temp_prompt
    if file_type in ["image", "single-image", "multi-image"]:
        multimodal_inputs = processor(images=input_images, text=full_prompt, return_tensors="pt", padding=True).to(device, dtype=torch.float32)
    elif file_type == "video":
        multimodal_inputs = processor(videos=input_images[0], text=full_prompt, return_tensors="pt", padding=True).to(device, dtype=torch.float32)
        
    
    tokens = multimodal_inputs
    
    targets = processor.tokenizer(labels[0], add_special_tokens=False,
                    return_tensors="pt", padding=True, max_length=multimodal_inputs["input_ids"].size(1))["input_ids"]

    labels_ids = torch.full_like(multimodal_inputs["input_ids"], -100)
    labels_ids[:, -targets.size(1):] = targets
    tokens["labels"] = labels_ids
    tokens = tokens.to(device)
    return tokens