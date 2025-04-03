import sys






sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import os
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import random

from .generate_caa_hparam import CAAHyperParams

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")




def generate_caa_vectors(hparams:CAAHyperParams, dataset, model = None):
    from ...models.get_model import get_model
    from ...datasets.caa_data import get_tokens_for_caa
    # Whether to delete the model after the function
    del_model = True
    args = hparams
    if model is None:
        model, tokenizer = get_model(hparams)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams

    pos_activations = dict([(layer, []) for layer in args.layers])
    neg_activations = dict([(layer, []) for layer in args.layers])


    pos_tokens_list, neg_tokens_list = get_tokens_for_caa(dataset, tokenizer, hparams)

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
            if args.multiple_choice == True:
                p_activations = p_activations[0, -2, :].detach().cpu()
            else:
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            pos_activations[layer].append(p_activations)

        model.reset_all()
        model.get_logits(n_tokens)
        
        ques_tokens_len = n_tokens_dict["ques_tokens_len"]
        for layer in args.layers:
            n_activations = model.get_last_activations(layer)
            if args.multiple_choice == True:
                n_activations = n_activations[0, -2, :].detach().cpu()
            else:
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach().cpu()
            neg_activations[layer].append(n_activations)
            
    if hparams.save_vectors is True:
        if args.multiple_choice == True:
            steer_vector_output_dir = os.path.join(
                args.steer_vector_output_dir, "caa_vector_multiple_choice"
            )
        else:
            steer_vector_output_dir = os.path.join(
                args.steer_vector_output_dir, "caa_vector"
            )

        if not os.path.exists(steer_vector_output_dir):
            os.makedirs(steer_vector_output_dir)

    try:
        vectors = {}
        for layer in args.layers:
            all_pos_layer = torch.stack(pos_activations[layer])
            all_neg_layer = torch.stack(neg_activations[layer])
            vec = (all_pos_layer - all_neg_layer).mean(dim=0)
            if hparams.save_vectors is True:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/16t/xzwnlp/SaeEdit/git/shae/data/safety/safeedit",
    )
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/20t/msy/models/gemma-2-9b-base")
    parser.add_argument("--multiple_choice", action="store_true", default=False)
    parser.add_argument("--steer_vector_output_dir", type=str, default="../")
    
    args = parser.parse_args()

    if args.hparams_path:
        hparams = CAAHyperParams.from_hparams(args.hparams_path)
    else:
        hparams = CAAHyperParams(
            layers=args.layers,
            data_path=args.data_path,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            multiple_choice=args.multiple_choice,
            steer_vector_output_dir=args.steer_vector_output_dir,
        )

    generate_caa_vectors(hparams)
