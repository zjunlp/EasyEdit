from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from .qlora_hparams import QLoRAHyperParams

def apply_qlora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: QLoRAHyperParams,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes using QLoRA.
    """
    edited_model = execute_qlora(model, tok, requests, hparams)
    return edited_model, {}

def execute_qlora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: QLoRAHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the QLoRA update algorithm for the specified requests
    """
    
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=hparams.lora_r,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=hparams.target_modules
    )
    model = get_peft_model(model, peft_config)

    # Training setup
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # hparams.device = 1
    device = torch.device(f"cuda:{hparams.device}")
    model.to(device)

    # Prepare data
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    # Training loop
    model.train()
    for step in range(hparams.num_steps):
        total_loss = 0
        for text, target in zip(texts, targets):
            inputs = tok(text, return_tensors="pt", max_length=hparams.max_length, truncation=True, padding="max_length").to(device)
            target_ids = tok(target, return_tensors="pt", max_length=hparams.max_length, truncation=True, padding="max_length")["input_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{hparams.num_steps}, Average Loss: {total_loss / len(texts):.4f}")

    model.eval()
    return model

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count