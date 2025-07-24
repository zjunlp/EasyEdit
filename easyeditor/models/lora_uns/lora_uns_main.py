from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from .lora_uns_hparams import LoRA_uns_HyperParams

def apply_lora_uns_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        hparams: LoRA_uns_HyperParams,
        batch_data: list,  # AnyEdit：[{'question': '...', 'answer': '...'}, ...]
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    edited_model = execute_lora_uns(model, tok, hparams, batch_data, keep_original_weight)

    return edited_model, weights_copy


def execute_lora_uns(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        hparams: LoRA_uns_HyperParams,
        batch_data: list,
        keep_original_weight=False,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the Lora update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError
    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model
    else:
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    if hasattr(peft_model, 'print_trainable_parameters'):
        peft_model.print_trainable_parameters()
    batch_data = deepcopy(batch_data)
    
    device = torch.device(f'cuda:{hparams.device}')
    # Define inputs
    texts = [data["question"] for data in batch_data]
    targets = [data["answer"] for data in batch_data]

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
                chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            mask_token = -100
            opt.zero_grad()
            full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
            prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
            num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
            tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
            bs = tokens["input_ids"].shape[0]
            tokens["labels"] = tokens["input_ids"].clone()
            num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
            for i in range(len(txt)):
                tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
            tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
            tokens = tokens.to(device)
            pred = peft_model(**tokens)
            loss = pred.loss

            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            loss.backward()
            opt.step()

        print(f"Total loss {loss_meter.avg}")

    return peft_model

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


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk
