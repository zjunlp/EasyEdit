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

def generate_prompt(model, tokenizer, concept, hparams):
    from ...vector_appliers.utils.templates import build_model_input
    with torch.no_grad():
        generate_prompt_params = hparams.generate_prompt_params
        if 'pad_token_id' not in generate_prompt_params:
            generate_prompt_params['pad_token_id'] = tokenizer.eos_token_id
        prompt = build_model_input( T_GENERATE_STEERING_PROMPT % concept , 
                                   tokenizer,  system_prompt='',
                                   use_chat_template = hparams.use_chat_template)
        model_inputs =  tokenizer([prompt], return_tensors="pt").to(model.device)
        generated_ids = model.model.generate(**model_inputs, **generate_prompt_params)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip('"')
    return response 

def generate_prompt_vllm(model, VLLM_model, concept, hparams):
    from ...vector_appliers.utils.templates import build_model_input
    from vllm import SamplingParams
    
    tokenizer = VLLM_model.llm_engine.tokenizer
    
    with torch.no_grad():
        generate_prompt_params = hparams.generate_prompt_params
        prompt = build_model_input( T_GENERATE_STEERING_PROMPT % concept , 
                                   tokenizer,  system_prompt='',
                                   use_chat_template = hparams.use_chat_template)
        vllm_sampling_params = SamplingParams(
            temperature=generate_prompt_params.get("temperature", 0.0),
            top_p=generate_prompt_params.get("top_p", 1.0),
            max_tokens=generate_prompt_params.get("max_new_tokens", 100),
            top_k=generate_prompt_params.get("top_k", -1),
            repetition_penalty=generate_prompt_params.get("repetition_penalty", 1.0),
            presence_penalty=generate_prompt_params.get("presence_penalty", 0.0),
            frequency_penalty=generate_prompt_params.get("frequency_penalty", 0.0),
            stop=generate_prompt_params.get("stop", None),
            stop_token_ids=generate_prompt_params.get("stop_token_ids", None),
        )
        outputs = VLLM_model.generate([prompt], sampling_params=vllm_sampling_params)
        response = outputs[0].outputs[0].text.strip('"')
        
    return response


def apply_prompt(hparams: ApplyPromptHyperParams, pipeline=None, tokenizer=None):
    from ...models.get_model import get_model
    VLLM_model = None  
    
    if pipeline is None:
        if getattr(hparams, 'vllm_enable', False):
            model, VLLM_model = get_model(hparams)
        else:
            model, tokenizer = get_model(hparams)
    else:
        model = pipeline
        
        if tokenizer is not None:
            tokenizer = tokenizer
        elif hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer provided and model does not have a tokenizer attribute.")

    prompt = hparams.prompt
    if hparams.generate_prompt_params:
        if getattr(hparams, 'vllm_enable', False):
            if VLLM_model is not None:
                prompt = generate_prompt_vllm(model, VLLM_model, prompt, hparams)
            else:
                raise ValueError("VLLM_model is None but vllm_enable is True")
        else:
            prompt = generate_prompt(model, tokenizer, prompt, hparams)

    setattr(model, 'prompt', prompt)
    if hparams.vllm_enable:
        return model, VLLM_model
    else:
        return model, tokenizer
        
