import json
import torch
import sys
from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig
import random
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from steer.datasets.dataset_loader import DatasetLoader
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.evaluate.evaluate import Evaluator

def load_axbench_datasets() -> list:
    print(f"Loading 500 concepts from axbench...")
    dataset_loader = DatasetLoader()
    train_data = dataset_loader.load_file('axbench_train', 'train')
    
    # train_data is now a list of tuples: [(concept_id, concept_data_list), ...]
    concept_grouped_data = {}
    for concept_id, concept_data_list in train_data:
        concept_grouped_data[concept_id] = concept_data_list

    eval_data = dataset_loader.load_file('axbench_test', 'generation')  # The Apacha-Eval dataset
    return concept_grouped_data, eval_data

@hydra.main(version_base='1.2', config_path='../hparams/Steer/experiment_hparams/prism_experiment/axbench', config_name='axbench_config.yaml')
def main(top_cfg: DictConfig):
    print("Global Config:", top_cfg, "\n")

    # Initialize and load
    train_datasets, eval_datasets = load_axbench_datasets()
    all_evaluation_results = []
    all_generation_results = []  
    
    vector_generator = BaseVectorGenerator(top_cfg)
    vector_applier = None
    
    # Get layers from config for metadata
    save_vectors = False
    layers = None
    if 'caa' in vector_generator.hparams_dict:
        save_vectors = False
        caa_output_dir = os.path.join(top_cfg.steer_vector_output_dirs[0], "axbench", f"{top_cfg.axbench_output_dir_name}")
        os.makedirs(caa_output_dir, exist_ok=True)
        layers = vector_generator.hparams_dict['caa'].layers
    
    if 'reps' in vector_generator.hparams_dict:
        save_vectors = False
        reps_output_dir = os.path.join(top_cfg.steer_vector_output_dirs[0], "axbench", f"{top_cfg.axbench_output_dir_name}")
        os.makedirs(reps_output_dir, exist_ok=True)
        layers = vector_generator.hparams_dict['reps'].layers

    if 'sft' in vector_generator.hparams_dict:
        save_vectors = False
        sft_output_dir = os.path.join(top_cfg.steer_vector_output_dirs[0], "axbench", f"{top_cfg.axbench_output_dir_name}")
        os.makedirs(sft_output_dir, exist_ok=True)
        layers = vector_generator.hparams_dict['sft'].layers

    if 'prism' in vector_generator.hparams_dict:
        save_vectors = False
        prism_output_dir = os.path.join(top_cfg.steer_vector_output_dirs[0], "axbench", f"{top_cfg.axbench_output_dir_name}")
        os.makedirs(prism_output_dir, exist_ok=True)
        layers = vector_generator.hparams_dict['prism'].layers

    for i, concept_train_data in train_datasets.items():
        if i >= 10:
            continue
        print(f"Processing concept {i} with {len(concept_train_data)} items")
        train_dataset_for_concept = {
            f'axbench_concept_{i}': concept_train_data
        }
        vectors = vector_generator.generate_vectors(train_dataset_for_concept)[f'axbench_concept_{i}']
        print(f"Generated vectors for concept {i}: {vectors}")
        # Save CAA vectors following RePS structure
        if save_vectors and layers:
            for layer in layers:
                layer_key = f"layer_{layer}"
                # Check if this layer exists in vectors dict
                if 'caa' in vectors and layer_key in vectors['caa']:
                    # vectors structure: {layer_key: {'caa': tensor}}
                    if 'caa' in vectors and isinstance(vectors['caa'], dict) and layer_key in vectors['caa']:
                        vec = vectors['caa'][layer_key]  # Get CAA vector for this layer
                    elif layer_key in vectors:
                        # vectors structure: {layer_key: tensor} (direct tensor)
                        vec = vectors[layer_key]
                    else:
                        raise KeyError(f"Layer key '{layer_key}' not found in vectors dictionary.")
                    
                    # Ensure vec is on CPU and detached
                    if hasattr(vec, 'cpu'):
                        vec = vec.cpu().detach()
                    
                    # Load existing vectors if file exists, otherwise create new list
                    vector_filename = f"layer_{layer}.pt"
                    vector_filepath = os.path.join(caa_output_dir, vector_filename)
                    
                    if os.path.exists(vector_filepath):
                        existing_vectors = torch.load(vector_filepath)
                        # Convert to list if it's a tensor
                        if isinstance(existing_vectors, torch.Tensor):
                            existing_vectors = [existing_vectors[j] for j in range(existing_vectors.shape[0])]
                        elif not isinstance(existing_vectors, list):
                            existing_vectors = [existing_vectors]
                    else:
                        existing_vectors = []
                    
                    # Append new vector
                    existing_vectors.append(vec)
                    
                    # Save updated vectors as stacked tensor
                    combined_vectors = torch.stack(existing_vectors, dim=0)
                    torch.save(combined_vectors, vector_filepath)
                    
                    # Save metadata for this concept
                    metadata = {
                        "concept_id": int(i),
                        "concept": concept_train_data[0]['output_concept'],
                        "ref": f"vector_index_{len(existing_vectors)-1}"  # Index in the stacked tensor
                    }
                    
                    metadata_file = os.path.join(caa_output_dir, f"metadata_layer_{layer}.jsonl")
                    with open(metadata_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata) + '\n')
                    
                    print(f"[INFO] Saved CAA vector for concept {i} to layer_{layer}.pt (total: {len(existing_vectors)} vectors)")

                if 'reps' in vectors and layer_key in vectors['reps']:
                    # vectors structure: {layer_key: {'reps_vector': tensor}}
                    if 'reps' in vectors and isinstance(vectors['reps'], dict) and layer_key in vectors['reps']:
                        data_state = vectors['reps'][layer_key]  # Get reps vector for this layer
                    elif layer_key in vectors:
                        # vectors structure: {layer_key: tensor} (direct tensor)
                        data_state = vectors[layer_key]
                    else:
                        raise KeyError(f"Layer key '{layer_key}' not found in vectors dictionary.")
                    
                    # Ensure vec is on CPU and detached
                    if hasattr(data_state, 'cpu'):
                        data_state = data_state.cpu().detach()
                    else:
                        for k in data_state:
                            if hasattr(data_state[k], 'cpu'):
                                data_state[k] = data_state[k].cpu().detach()
                    
                    # Load existing vectors if file exists, otherwise create new list
                    vector_filename = f"layer_{layer}.pt"
                    vector_filepath = os.path.join(reps_output_dir, vector_filename)

                    if os.path.exists(vector_filepath):
                        existing_vectors = torch.load(vector_filepath)
                        # Convert to list if it's a tensor
                        if isinstance(existing_vectors, torch.Tensor):
                            existing_vectors = [existing_vectors[j] for j in range(existing_vectors.shape[0])]
                        elif not isinstance(existing_vectors, list):
                            existing_vectors = [existing_vectors]
                    else:
                        existing_vectors = []
                    
                    # Append new vector
                    if top_cfg.intervention_method=="vector":
                        existing_vectors.append(data_state)
                        combined_vectors = torch.stack(existing_vectors, dim=0)
                        torch.save(combined_vectors, vector_filepath)
                    else:
                        torch.save(data_state, vector_filepath)
                    
                    # Save metadata for this concept
                    metadata = {
                        "concept_id": int(i),
                        "concept": concept_train_data[0]['output_concept'],
                        "ref": f"vector_index_{len(existing_vectors)-1}"  # Index in the stacked tensor
                    }
                    
                    metadata_file = os.path.join(reps_output_dir, f"metadata_layer_{layer}.jsonl")
                    with open(metadata_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata) + '\n')
                    
                    print(f"[INFO] Saved reps_{top_cfg.intervention_method} for concept {i} to layer_{layer}.pt (total: {len(existing_vectors)})")

                if 'sft' in vectors and layer_key in vectors['sft']:
                    # vectors structure: {layer_key: {'sft': tensor}}
                    if 'sft' in vectors and isinstance(vectors['sft'], dict) and layer_key in vectors['sft']:
                        data_state = vectors['sft'][layer_key]  # Get sft vector for this layer
                    elif layer_key in vectors:
                        # vectors structure: {layer_key: tensor} (direct tensor)
                        data_state = vectors[layer_key]
                    else:
                        raise KeyError(f"Layer key '{layer_key}' not found in vectors dictionary.")
                    
                    # Ensure vec is on CPU and detached
                    if hasattr(data_state, 'cpu'):
                        data_state = data_state.cpu().detach()
                    else:
                        for k in data_state:
                            if hasattr(data_state[k], 'cpu'):
                                data_state[k] = data_state[k].cpu().detach()
                    
                    # Load existing vectors if file exists, otherwise create new list
                    vector_filename = f"layer_{layer}.pt"
                    vector_filepath = os.path.join(sft_output_dir, vector_filename)

                    if os.path.exists(vector_filepath):
                        existing_vectors = torch.load(vector_filepath)
                        # Convert to list if it's a tensor
                        if isinstance(existing_vectors, torch.Tensor):
                            existing_vectors = [existing_vectors[j] for j in range(existing_vectors.shape[0])]
                        elif not isinstance(existing_vectors, list):
                            existing_vectors = [existing_vectors]
                    else:
                        existing_vectors = []
                    
                    # Append new vector
                    if top_cfg.intervention_method=="vector":
                        existing_vectors.append(data_state)
                        combined_vectors = torch.stack(existing_vectors, dim=0)
                        torch.save(combined_vectors, vector_filepath)
                    else:
                        torch.save(data_state, vector_filepath)
                    
                    # Save metadata for this concept
                    metadata = {
                        "concept_id": int(i),
                        "concept": concept_train_data[0]['output_concept'],
                        "ref": f"vector_index_{len(existing_vectors)-1}"  # Index in the stacked tensor
                    }
                    
                    metadata_file = os.path.join(sft_output_dir, f"metadata_layer_{layer}.jsonl")
                    with open(metadata_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata) + '\n')
                    
                    print(f"[INFO] Saved sft_{top_cfg.intervention_method} for concept {i} to layer_{layer}.pt (total: {len(existing_vectors)})")

                if 'prism' in vectors and layer_key in vectors['prism']:
                    # vectors structure: {layer_key: {'prism': tensor}}
                    if 'prism' in vectors and isinstance(vectors['prism'], dict) and layer_key in vectors['prism']:
                        data_state = vectors['prism'][layer_key]  # Get prism vector for this layer
                    elif layer_key in vectors:
                        # vectors structure: {layer_key: tensor} (direct tensor)
                        data_state = vectors[layer_key]
                    else:
                        raise KeyError(f"Layer key '{layer_key}' not found in vectors dictionary.")
                    
                    # Ensure vec is on CPU and detached
                    if hasattr(data_state, 'cpu'):
                        data_state = data_state.cpu().detach()
                    else:
                        for k in data_state:
                            if hasattr(data_state[k], 'cpu'):
                                data_state[k] = data_state[k].cpu().detach()
                    
                    # Load existing vectors if file exists, otherwise create new list
                    vector_filename = f"layer_{layer}.pt"
                    vector_filepath = os.path.join(prism_output_dir, vector_filename)

                    if os.path.exists(vector_filepath):
                        existing_vectors = torch.load(vector_filepath)
                        # Convert to list if it's a tensor
                        if isinstance(existing_vectors, torch.Tensor):
                            existing_vectors = [existing_vectors[j] for j in range(existing_vectors.shape[0])]
                        elif not isinstance(existing_vectors, list):
                            existing_vectors = [existing_vectors]
                    else:
                        existing_vectors = []
                    
                    # Append new vector
                    if top_cfg.intervention_method=="vector":
                        existing_vectors.append(data_state)
                        combined_vectors = torch.stack(existing_vectors, dim=0)
                        torch.save(combined_vectors, vector_filepath)
                    else:
                        torch.save(data_state, vector_filepath)
                    
                    # Save metadata for this concept
                    metadata = {
                        "concept_id": int(i),
                        "concept": concept_train_data[0]['output_concept'],
                        "ref": f"vector_index_{len(existing_vectors)-1}"  # Index in the stacked tensor
                    }
                    
                    metadata_file = os.path.join(prism_output_dir, f"metadata_layer_{layer}.jsonl")
                    with open(metadata_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata) + '\n')
                    
                    print(f"[INFO] Saved prism_{top_cfg.intervention_method} for concept {i} to layer_{layer}.pt (total: {len(existing_vectors)})")


if __name__ == '__main__':
    main()

