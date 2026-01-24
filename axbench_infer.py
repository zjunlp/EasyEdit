import json
import torch

import pandas as pd
import hydra
from omegaconf import DictConfig
import random
from steer.datasets.dataset_loader import DatasetLoader
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.evaluate.evaluate import Evaluator
import os

def load_axbench_datasets() -> list:
    print(f"Loading 500 concepts from axbench...")
    dataset_loader = DatasetLoader()

    eval_data = dataset_loader.load_file('axbench_test', 'generation')  # The Apacha-Eval dataset
    concept_grouped_data = {}
    for concept_id, concept_data_list in eval_data:
        concept_grouped_data[concept_id] = concept_data_list

    return concept_grouped_data

@hydra.main(version_base='1.2', config_path='./hparams/Steer/acl_experiment/axbench', config_name='axbench_config.yaml')
def main(top_cfg: DictConfig):
    print("Global Config:", top_cfg, "\n")

    # Initialize and load
    eval_datasets = load_axbench_datasets()

    # all_evaluation_results = []
    all_generation_results = []  

    vector_applier = None

    eval_args = {"mode": 'direct', "save_results": False, "eval_methods": ["llm"], "llm_model": "gpt-4o-mini-2024-07-18" , 'save_path': top_cfg.generation_output_dir}
    # evaluator = Evaluator(**eval_args)

    for i, concept_eval_data in eval_datasets.items():
        if i >= 10:
            continue
        print(f"Processing concept {i} with {len(concept_eval_data)} items")

        # Save RePS vectors
        if vector_applier is None:
            vector_applier = BaseVectorApplier(top_cfg)
        
        if "prompt" in vector_applier.hparams_dict:
            vector_applier.hparams_dict['prompt'].prompt = concept_eval_data[0]['output_concept']
        # vector_applier.apply_vectors(vectors)   # Different from axbench, the steering factor is fixed for each concept
        
        elif "reps" in vector_applier.hparams_dict:
            m = vector_applier.hparams_dict['reps'].multipliers[0]
            vector_applier.hparams_dict['sft'].steer_vector_load_dir=f"vectors/{top_cfg.model_name}/{top_cfg.method}/axbench_concept_{i}/{top_cfg.vector_name}/"
            print(f"Updated RePS vector applier config: {vector_applier.hparams_dict['reps']}")

        elif "caa" in vector_applier.hparams_dict:
            m = vector_applier.hparams_dict['caa'].multipliers[0]
            # vector_applier.hparams_dict['our'].steer_vector_load_dir=f"vectors/gemma-2-9b-it/axbench_concept_{i}/our_vector/"
            vector_applier.hparams_dict['sft'].steer_vector_load_dir=f"vectors/{top_cfg.model_name}/{top_cfg.method}/axbench_concept_{i}/{top_cfg.vector_name}/"
            print(f"Updated CAA vector applier config: {vector_applier.hparams_dict['caa']}")


        elif "sft" in vector_applier.hparams_dict:
            m = vector_applier.hparams_dict['sft'].multipliers[0]
            # vector_applier.hparams_dict['our'].steer_vector_load_dir=f"vectors/gemma-2-9b-it/axbench_concept_{i}/our_vector/"
            vector_applier.hparams_dict['sft'].steer_vector_load_dir=f"vectors/{top_cfg.model_name}/{top_cfg.method}/axbench_concept_{i}/{top_cfg.vector_name}/"
            print(f"Updated SFT vector applier config: {vector_applier.hparams_dict['sft']}")
        
        elif "prism" in vector_applier.hparams_dict:
            m = vector_applier.hparams_dict['prism'].multipliers[0]
            # vector_applier.hparams_dict['our'].steer_vector_load_dir=f"vectors/gemma-2-9b-it/axbench_concept_{i}/our_vector/"
            vector_applier.hparams_dict['prism'].steer_vector_load_dir=f"vectors/{top_cfg.model_name}/{top_cfg.method}/axbench_concept_{i}/{top_cfg.vector_name}/"
            print(f"Updated PRISM vector applier config: {vector_applier.hparams_dict['prism']}")

        else:
            raise ValueError("No valid vector applier configuration found.")

        vector_applier.apply_vectors()   # Different from axbench, the steering factor is fixed for each concept

        # Generate results using the vector applier
        generated_results = vector_applier.generate(
           {
            'axbench_eval': concept_eval_data
           },
           save_results=False
        )
        
        save_data = {
            'concept_id': i,
            'concept_name': concept_eval_data[0]['output_concept'],
            'generated_results': generated_results,
            'eval_data': concept_eval_data,
            'multiplier': m,
        }
        all_generation_results.append(save_data)
        
        vector_applier.model.reset_all()

    print("Saving all generation results to file...")

    if "reps" in vector_applier.hparams_dict:
        m = vector_applier.hparams_dict['reps'].multipliers[0]
    elif "sft" in vector_applier.hparams_dict:
        m = vector_applier.hparams_dict['sft'].multipliers[0]
    elif "caa" in vector_applier.hparams_dict:
        m = vector_applier.hparams_dict['caa'].multipliers[0]
    generation_file_path = f"{eval_args['save_path']}/all_generation_results_m{m}.json"
    os.makedirs(os.path.dirname(generation_file_path), exist_ok=True)
    with open(generation_file_path, "w", encoding="utf-8") as f:
        json.dump(all_generation_results, f, indent=2, ensure_ascii=False)
    print(f"All results saved to files: {generation_file_path}")


if __name__ == '__main__':
    main()

