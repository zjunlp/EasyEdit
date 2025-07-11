import sys

import torch.utils.data.dataset






sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")
import os
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import random
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, TrainingArguments, set_seed
import pyreft
import matplotlib.pyplot as plt
from copy import deepcopy

from .generate_LoReFT_hparams import LoReFTHyperParams
IGNORE_INDEX  = -100

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
def save_interventions(reft_model, dump_dir,model_name):
    proj_weights = []
    source_weights = []
    source_biases = []
    intervention_names = []
    for intervention_name, intervention in reft_model.interventions.items():
        intervention_names.append(intervention_name)
        intervention_state_dict = intervention.state_dict()
        proj_weight = intervention_state_dict["rotate_layer"] # [embed_dim, low_rank_dimension]
        source_weight = intervention_state_dict["weight"].T # [embed_dim, low_rank_dimension]
        source_bias = intervention_state_dict["bias"] # [low_rank_dimension]
        proj_weights.append(proj_weight)
        source_weights.append(source_weight)
        source_biases.append(source_bias)
    weight_file = dump_dir / f"{model_name}_weight.pt"
    if weight_file.exists():
        existing_weight = torch.load(weight_file)
        for i, intervention_name in enumerate(intervention_names):
            existing_weight[f"{intervention_name}.proj_weight"] = torch.cat(
                [existing_weight[f"{intervention_name}.proj_weight"], proj_weights[i].cpu().unsqueeze(dim=0)], dim=0)
            existing_weight[f"{intervention_name}.source_weight"] = torch.cat(
                [existing_weight[f"{intervention_name}.source_weight"], source_weights[i].cpu().unsqueeze(dim=0)], dim=0)
    else:
        existing_weight = {}
        for i, intervention_name in enumerate(intervention_names):
            existing_weight[f"{intervention_name}.proj_weight"] = proj_weights[i].cpu().unsqueeze(dim=0)
            existing_weight[f"{intervention_name}.source_weight"] = source_weights[i].cpu().unsqueeze(dim=0)
    torch.save(existing_weight, weight_file)
    bias_file = dump_dir / f"{model_name}_bias.pt"
    if bias_file.exists():
        existing_bias = torch.load(bias_file)
        for i, intervention_name in enumerate(intervention_names):
            existing_bias[f"{intervention_name}.bias"] = torch.cat(
                [existing_bias[f"{intervention_name}.bias"], source_biases[i].cpu().unsqueeze(dim=0)], dim=0)
    else:
        existing_bias = {}
        for i, intervention_name in enumerate(intervention_names):
            existing_bias[f"{intervention_name}.bias"] = source_biases[i].cpu().unsqueeze(dim=0)
    torch.save(existing_bias, bias_file)
    
def generate_LoReFT_vectors(hparams:LoReFTHyperParams, dataset, model = None):
    from ...models.get_model import get_model
    from ...datasets.loreft_data import load_loreft_data
    from transformers import get_scheduler
    import transformers
    from pathlib import Path
    del_model = True
    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams
    system_prompt = "" if hparams.system_prompt is None else hparams.system_prompt
    use_chat_template = True if hparams.use_chat_template is None else hparams.use_chat_template
    subset = None if hparams.subset is None else hparams.subset
    tokenizer.model_max_length = hparams.max_length
    model.model.eval()
    device = hparams.device
    model.model.to(device)
    train_data = load_loreft_data(dataset,subset,tokenizer,system_prompt,use_chat_template)
    torch_dtype = model.torch_dtype
    batch_size = hparams.batch_size
    low_rank_dimension = hparams.low_rank_dimension
    if hparams.steer_vector_output_dir is None:
        assert "Need Steer Vector Output Dir"
    position = hparams.position
    reft_layers = hparams.reft_layers
    num_interventions = len(reft_layers)
    data_module = pyreft.make_multiple_position_supervised_data_module(
        tokenizer=tokenizer,model=model.model,
        inputs = [item["prompt"] for item in train_data],
        outputs= [item["output"] for item in train_data],
        positions = position,
        num_interventions= num_interventions,
        nonstop=True,
        share_weights=True
    )

    train_dataloader = DataLoader(
        data_module["train_dataset"],shuffle=True,
        batch_size=batch_size,collate_fn=data_module["data_collator"]
    )
    # make_model
    intervention_cls = pyreft.LoreftIntervention

    reft_config = pyreft.ReftConfig(representations=[{
        "layer": l, "component": "block_output",
        "low_rank_dimension": low_rank_dimension,
        "intervention": intervention_cls(embed_dim=model.model.config.hidden_size,
        low_rank_dimension=low_rank_dimension,dtype = torch_dtype)} for l in reft_layers])
    reft_model = pyreft.get_reft_model(model.model, reft_config)
    lr = hparams.lr
    weight_decay = hparams.weight_decay
    optimizer = torch.optim.AdamW(
        reft_model.parameters(), lr=lr,weight_decay=weight_decay,
        betas=(0.9,0.999),eps = 1e-8
    )
    n_epochs= hparams.n_epochs
    gradient_accumulation_steps = 1 if hparams.gradient_accumulation_steps is None else hparams.gradient_accumulation_steps
    num_training_steps = n_epochs * max(1, len(train_dataloader))//gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,num_training_steps=num_training_steps
    )
    progress_bar, curr_step = tqdm(range(num_training_steps), leave=True), 0

    reft_model.print_trainable_parameters()
    losses = []
    for epoch in range(n_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = {k : v.to(device) for k,v in batch.items()}
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            _, cf_outputs = reft_model.forward(
                base = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },unit_locations=unit_locations,
                labels=inputs["labels"],
                use_cache=False 
            )
            loss = cf_outputs.loss.mean()
            loss.backward()
            loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            if(step + 1) % gradient_accumulation_steps == 0 or ( step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(reft_model.parameters(), 1.0)
                curr_step += 1
                curr_lr = get_lr(optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                losses.append(loss.item())
                progress_bar.set_description(
                    "lr %.6f || loss %.6f " % (curr_lr, loss))
    progress_bar.close()
    save_directory = os.path.join(hparams.steer_vector_output_dir, hparams.alg_name + '_vector')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_interventions(reft_model,Path(save_directory),"loreft")
    if del_model:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


