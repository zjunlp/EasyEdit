import sys




sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import os
import torch
import warnings
from tqdm import tqdm
from dotenv import load_dotenv

from .generate_vector_prompt_hparam import VectorPromptHyperParams

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.tokenize(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens


def generate_vector_prompt_vectors(hparams:VectorPromptHyperParams,dataset, model = None):
    

    from ...models.get_model import get_model
    from ...datasets.caa_data import get_tokens_for_vector_prompt
    args = hparams
    del_model = True
    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams

    if args.generate_prompt_params:
        from ...vector_appliers.prompt.apply_prompt import generate_prompt
        cache_memory = {}
        for item in dataset:
            prompt = item.get('prompt', hparams.prompt).strip() 
            if  prompt not in cache_memory:
                gen_prompt = generate_prompt(model, tokenizer,prompt, args.generate_prompt_params)
                cache_memory[ prompt ] = gen_prompt 
            item["prompt"] = cache_memory[ prompt ]
    # for data in dataset:
    #     print(data)
    pos_activations = dict([(layer, []) for layer in args.layers])
    neg_activations = dict([(layer, []) for layer in args.layers])
    pos_tokens_list, neg_tokens_list = get_tokens_for_vector_prompt(dataset, tokenizer, hparams)


    for p_tokens_dict, n_tokens_dict in tqdm(
        zip(pos_tokens_list, neg_tokens_list),
        total=len(pos_tokens_list),
        desc="Processing prompts",
    ):
        p_tokens = p_tokens_dict["pos_tokens"]
        n_tokens = n_tokens_dict["neg_tokens"]
        ques_tokens_len = p_tokens_dict["ques_tokens_len"]
        model.reset_all()
        model.get_logits(p_tokens)

        for layer in args.layers:
            p_activations = model.get_last_activations(layer)
            # mean the activation over all answer tokens
            p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            pos_activations[layer].append(p_activations)

        model.reset_all()
        model.get_logits(n_tokens)
        ques_tokens_len = n_tokens_dict["ques_tokens_len"]
        for layer in tqdm(args.layers, desc="Saving steer vectors"):
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            neg_activations[layer].append(n_activations)
    if args.generate_prompt_params:
        setattr(model, 'generate_prompts', cache_memory)
    try:
        vectors = {}
        for layer in args.layers:
            all_pos_layer = torch.stack(pos_activations[layer])
            all_neg_layer = torch.stack(neg_activations[layer])
            vec = (all_pos_layer - all_neg_layer).mean(dim=0)
            if hparams.save_vectors is True:
                steer_vector_output_dir = os.path.join(
                    args.steer_vector_output_dir, "vector_prompt_vector"
                )

                if not os.path.exists(steer_vector_output_dir):
                    os.makedirs(steer_vector_output_dir)
                torch.save(
                    vec,
                    os.path.join(steer_vector_output_dir, f"layer_{layer}.pt"),
                )
            vectors[f"layer_{layer}"] = vec
    finally:
        if del_model:
            model.model.to('cpu')
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return vectors
