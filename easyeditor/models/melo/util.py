import transformers
import torch
import os
import numpy as np
import datetime
import struct
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import hydra

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
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value

def scr():
    base_dir = hydra.utils.get_original_cwd()
    if os.path.exists(os.path.join(base_dir,"scr-ssd")):
        scr_dir = os.path.join(base_dir,"scr-ssd")
    else:
        scr_dir = os.path.join(base_dir,"scr")

    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)

    return scr_dir
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


def tokenize_gpt(batch, tokenizer, device, test=False):
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100  # ignore_index of CrossEntropyLoss
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

        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token  # What is this doing?

    tokens = {f"{k1}": v1.to(device) for k1, v1 in tokens.items()}
    return tokens


def tokenize_qa(batch, tokenizer, device, **kwargs):
    input_sequences, output_sequences = batch["text"], batch["labels"]

    input_encoding = tokenizer(
        list(input_sequences),
        padding="longest",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    target_encoding = tokenizer(
        list(output_sequences),
        padding="longest",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    tokens = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    tokens = {f"{k1}": v1.to(device) for k1, v1 in tokens.items()}
    return tokens

def get_tokenizer(config):
    if config.task == 'hall':
        return tokenize_gpt
    else:
        return tokenize_qa






