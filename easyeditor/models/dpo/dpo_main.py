from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dpo_hparams import DPOHyperParams


def apply_dpo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
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

    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")
    # 保存参考模型（oracle model）
    ref_model = deepcopy(model).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    edited_model = execute_dpo(model, ref_model, tok, requests, hparams, keep_original_weight)

    return edited_model, weights_copy


def execute_dpo(
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the DPO algorithm for the specified updates.
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError
    if not keep_original_weight and hasattr(model, 'peft_config'):
        peft_model = model
    else:
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    # peft_model.print_trainable_parameters()

    requests = deepcopy(requests)
    for request in requests:
        print(
            f"Executing DPO algorithm for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    device = torch.device(f'cuda:{hparams.device}')
    peft_model.to(device)
    ref_model.to(device)
    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets_pos = [r["target_new"] for r in requests]  # Positive samples
    targets_neg = [r["target_neg"] for r in requests]  # Negative samples

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

        for txt_batch, tgt_pos_batch, tgt_neg_batch in zip(
                chunks(texts, hparams.batch_size),
                chunks(targets_pos, hparams.batch_size),
                chunks(targets_neg, hparams.batch_size),
        ):
            mask_token = -100
            opt.zero_grad()
            if 't5' in hparams.model_name.lower():
                # T5 model processing (if needed)
                pass  # Add T5-specific processing if required
            else:
                # Process positive samples
                full_prompt_pos = [f"{p} {l}" for p, l in zip(txt_batch, tgt_pos_batch)]
                tokens_pos = tok(full_prompt_pos, return_tensors="pt", padding=True, truncation=True)
                tokens_pos["labels"] = tokens_pos["input_ids"].clone()
                tokens_pos["labels"][tokens_pos["input_ids"] == tok.pad_token_id] = mask_token
                tokens_pos = tokens_pos.to(device)

                # Compute positive log probabilities using peft_model and ref_model
                with torch.no_grad():
                    ref_outputs_pos = ref_model(**tokens_pos)
                    ref_log_probs_pos = ref_outputs_pos.logits.log_softmax(-1)

                outputs_pos = peft_model(**tokens_pos)
                log_probs_pos = outputs_pos.logits.log_softmax(-1)
                # LoRA positive loss (standard language modeling loss on positive samples)
                lora_loss = outputs_pos.loss

                if hparams.alpha != 1:
                    # Process negative samples
                    full_prompt_neg = [f"{p} {l}" for p, l in zip(txt_batch, tgt_neg_batch)]
                    tokens_neg = tok(full_prompt_neg, return_tensors="pt", padding=True, truncation=True)
                    tokens_neg["labels"] = tokens_neg["input_ids"].clone()
                    tokens_neg["labels"][tokens_neg["input_ids"] == tok.pad_token_id] = mask_token
                    tokens_neg = tokens_neg.to(device)

                    # Compute negative log probabilities using peft_model and ref_model
                    with torch.no_grad():
                        ref_outputs_neg = ref_model(**tokens_neg)
                        ref_log_probs_neg = ref_outputs_neg.logits.log_softmax(-1)

                    outputs_neg = peft_model(**tokens_neg)
                    log_probs_neg = outputs_neg.logits.log_softmax(-1)

                    # DPO loss calculation (matching positive loss calculation)
                    beta = hparams.beta
                    dpo_advantage = beta * (
                        (log_probs_pos - ref_log_probs_pos).sum(-1) - 
                        (log_probs_neg - ref_log_probs_neg).sum(-1)
                    )
                    dpo_advantage = torch.tanh(dpo_advantage) # Loss smoothing. TODO: Please adjust the object depending on your goals.
                    dpo_loss = -torch.mean(torch.log(torch.sigmoid(dpo_advantage)))
                else:
                    dpo_loss = torch.tensor(0.0).to(device)

                # Combine losses
                loss = hparams.alpha * lora_loss + (1 - hparams.alpha) * dpo_loss
                print(f"lora_loss: {lora_loss.item()}, dpo_loss: {dpo_loss.item()}, total_loss: {loss.item()}")

            bs = len(txt_batch)
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
    for i in range(0, len(arr), n):
        yield arr[i:i + n]
