import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import get_model_device


def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    device = get_model_device(model)
    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to(device)

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()
