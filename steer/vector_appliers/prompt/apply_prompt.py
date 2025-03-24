from typing import Optional, Dict
from transformers import AutoTokenizer
from .apply_prompt_hparam import ApplyPromptHyperParams
import torch

T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""

def generate_prompt(model, tokenizer, concept, cfg):
    from ...utils import build_model_input
    with torch.no_grad():

        prompt = build_model_input( T_GENERATE_STEERING_PROMPT % concept , 
                                   tokenizer,  system_prompt='',
                                   use_chat_template = cfg['use_chat_template'])
        model_inputs =  tokenizer([prompt], return_tensors="pt").to(model.device)
        generated_ids = model.model.generate(**model_inputs, 
                       max_new_tokens=cfg['max_new_tokens'],
                       top_p=cfg['top_p'], 
                       temperature=cfg['temperature'],
                       do_sample=cfg['do_sample'],
                       pad_token_id = tokenizer.eos_token_id
                       )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip('"')
    return response 


def apply_prompt(hparams: ApplyPromptHyperParams, pipeline=None, tokenizer=None):
    from ...models.get_model import get_model
    if pipeline is None:
        model, tokenizer = get_model(hparams)
    else:
        model = pipeline
        tokenizer = tokenizer

    prompt = hparams.prompt
    if hparams.generate_prompt_params:
        prompt = generate_prompt(model, tokenizer, prompt, hparams.generate_prompt_params)

    setattr(model, 'prompt', prompt)
    return model
        
