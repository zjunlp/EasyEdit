import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os
from dotenv import load_dotenv

from .generate_sae_feature_hparam import SaeFeatureHyperParams
from typing import List

from .sae_utils import (
    clear_gpu_cache,
    load_sae_from_dir,
    load_gemma_2_sae,
)
load_dotenv()
# os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def prepare_input(tokenizer, prompts, device="cuda"):
    input_tokens = tokenizer.tokenize(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens

def get_sae_config(release, sae_id):
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    sae_directory = get_pretrained_saes_directory()

    # get the repo id and path to the SAE
    if release not in sae_directory:
        if "/" not in release:
            raise ValueError(
                f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
            )
    elif sae_id not in sae_directory[release].saes_map:
        # If using Gemma Scope and not the canonical release, give a hint to use it
        if (
            "gemma-scope" in release
            and "canonical" not in release
            and f"{release}-canonical" in sae_directory
        ):
            canonical_ids = list(
                sae_directory[release + "-canonical"].saes_map.keys()
            )
            # Shorten the lengthy string of valid IDs
            if len(canonical_ids) > 5:
                str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
            else:
                str_canonical_ids = str(canonical_ids)
            value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
        else:
            value_suffix = ""

        valid_ids = list(sae_directory[release].saes_map.keys())
        # Shorten the lengthy string of valid IDs
        if len(valid_ids) > 5:
            str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
        else:
            str_valid_ids = str(valid_ids)

        raise ValueError(
            f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
            + value_suffix
        )
    sae_info = sae_directory.get(release, None)
    hf_repo_id = sae_info.repo_id if sae_info is not None else release
    hf_path = sae_info.saes_map[sae_id] if sae_info is not None else sae_id
    config_overrides = sae_info.config_overrides if sae_info is not None else None
    neuronpedia_id = (
        sae_info.neuronpedia_id[sae_id] if sae_info is not None else None
    )

    return {
        "hf_repo_id": hf_repo_id,
        "hf_path": hf_path,
        "config_overrides": config_overrides,
        "neuronpedia_id": neuronpedia_id,
    }

def search_for_explanations(modelId, saeId, query, save_path=None):
    import requests

    # modelId, saeId = neuronpedia_id.split('/')
    url = "https://www.neuronpedia.org/api/explanation/search"
    payload = {
        "modelId": modelId,
        "layers": [saeId],
        "query": query
    }
    
    headers = {"Content-Type": "application/json", "X-Api-Key": os.environ.get("NP_API_KEY")}

    response = requests.post(url, json=payload, headers=headers)
    results = response.json()
    filtered_results = []
    for result in results['results']:
        filtered_result = {
            "modelId": result.get("modelId"),
            "layer": result.get("layer"),
            "index": result.get("index"),
            "description": result.get("description"),
            "explanationModelName": result.get("explanationModelName"),
            "typeName": result.get("typeName"),
            "cosine_similarity": result.get("cosine_similarity")
        }
        filtered_results.append(filtered_result)

    print(filtered_results)
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(filtered_results, f, indent=4)
    
    return filtered_results

def get_feature_description(modelId, saeId, position_ids, save_path=None):
    import requests
    import pandas as pd
    if save_path is not None and os.path.exists(save_path):
        description_df = pd.read_json(save_path)
    else:
        url = f"https://www.neuronpedia.org/api/explanation/export?modelId={modelId}&saeId={saeId}"
        headers = {"Content-Type": "application/json", "X-Api-Key": os.environ.get("NP_API_KEY")}

        response = requests.get(url, headers=headers)

        data = response.json()
        description_df = pd.DataFrame(data)
        # rename index to "feature"
        description_df.rename(columns={"index": "feature_id"}, inplace=True)
        description_df["feature_id"] = description_df["feature_id"].astype(int)
        description_df["description"] = description_df["description"].apply(
            lambda x: x.lower()
        )
        if save_path:
            description_df.to_json(save_path)

    result_dict = {k: description_df[description_df["feature_id"] == k].to_dict(orient='records') for k in position_ids}
    
    for k in position_ids:
        for item in result_dict[k]:
            print(f"Feature ID: {item['feature_id']}, Description: {item['description']}, ExplanationModelName: {item['explanationModelName']}")

    return result_dict

def generate_sae_feature_vectors(hparams:SaeFeatureHyperParams):
    args = hparams
    # print(args)
    device = hparams.device
    
    if args.sae_path is None:
        from sae_lens import SAE
        sae, sae_cfg, sparsity = SAE.from_pretrained(
            release=args.release,  # <- Release name
            sae_id=args.sae_id,  # <- SAE id (not always a hook point!)
            device=device,
        )
    else:
        assert str(args.layer) in args.sae_path, f"Sae doesnt match the layer!!!"
        if "gemma" in args.sae_path.lower():
            sae, sae_cfg, sparsity = load_gemma_2_sae(args.sae_path, device=device)
            if args.release and args.sae_id:
                sae_config = get_sae_config(args.release, args.sae_id)
                sae_cfg["neuronpedia_id"] = sae_config["neuronpedia_id"]
        else:
            sae, sae_cfg, sparsity = load_sae_from_dir(args.sae_path, device=device)
            if args.release and args.sae_id:
                sae_config = get_sae_config(args.release, args.sae_id)
                sae_cfg["neuronpedia_id"] = sae_config["neuronpedia_id"]

    if not os.path.exists(args.steer_vector_output_dir):
            os.makedirs(args.steer_vector_output_dir)

    position_ids = args.position_ids
    strengths = args.strengths
    sae_feature_vector = torch.zeros_like(sae.W_dec[position_ids[0]]).to(sae.device)
    for i,id in enumerate(position_ids):
        sae_feature_vector += strengths[i] * sae.W_dec[id]

    steer_vector_output_dir = os.path.join(
            args.steer_vector_output_dir, "sae_feature_vector"
    )

    if not os.path.exists(steer_vector_output_dir):
        os.makedirs(steer_vector_output_dir)
    steering_vector_path = os.path.join(steer_vector_output_dir, f"layer_{args.layer}.pt")
    if hparams.save_vectors is True:
        torch.save(
            sae_feature_vector.cpu(),
            steering_vector_path,
        )

    json_file = os.path.join(steer_vector_output_dir, f"sae_feature_config.json")

    # Construct a dictionary for the new entry
    new_entry = {
        "steering_vector_path": steering_vector_path,
        "layer": args.layer,
        "sae_path": args.sae_path,
        "release": args.release,
        "sae_id": args.sae_id,
        "strengths": strengths,
        "position_ids": position_ids,
    }

    # Attempts to retrieve and add features description to the dictionary.
    if sae_cfg["neuronpedia_id"]:
        try:
            modelId, saeId = sae_cfg["neuronpedia_id"].split('/')
            desc_data_path = os.path.join(steer_vector_output_dir, f"neuronpedia_description_{modelId}_{saeId}.json")
            description = get_feature_description(modelId, saeId, position_ids, desc_data_path)
            new_entry["description"] = description
        except Exception as e:
            print(f"Failed to get feature description: {e}")

    record_data = []
    # Reads the existing JSON file (if it exists) and updates it with the new entry.
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            record_data = json.load(f)

    found = False
    for entry in record_data:
        if entry.get("steering_vector_path") == steering_vector_path and entry.get("layer") == args.layer:
            entry.update(new_entry)
            found = True
            break

    if not found:
        record_data.append(new_entry)

    # Writes the updated data back to the JSON file.
    def convert_to_serializable(obj):
        if obj is None:
            return None
        return str(obj)

    with open(json_file, 'w') as f:
        json.dump(record_data, f, indent=4, default=convert_to_serializable)

    steering_vector_path = os.path.join(steer_vector_output_dir, f"layer_{args.layer}.pt")
    
    return  {f"layer_{args.layer}.pt" : sae_feature_vector}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument("--sae_paths", nargs="+", type=str, default=['/data2/xzwnlp/gemma-scope-9b-pt-res/layer_24/width_16k/average_l0_114'])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steer_vector_output_dir", type=str, default="../vectors/gemma-2-9b/sae_feature/")
    parser.add_argument("--position_ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--modelId", type=str, default="gemma-2-9b")
    parser.add_argument("--saeId", type=str, default="24-gemmascope-res-16k")
    
    args = parser.parse_args()

    if args.hparams_path:
        hparams = SaeFeatureHyperParams.from_hparams(args.hparams_path)
    else:
        hparams = SaeFeatureHyperParams(
            device=args.device,
            layers=args.layers,
            sae_path=args.sae_path,
            position_ids=args.position_ids,
            strengths=args.strengths,
            modelId=args.modelId,
            saeId=args.saeId,
            steer_vector_output_dir=args.steer_vector_output_dir,
        )

    generate_sae_feature_vectors(hparams)
