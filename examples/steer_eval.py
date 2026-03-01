import json

import pandas as pd
import hydra
from omegaconf import DictConfig
import random
from steer.datasets.dataset_loader import DatasetLoader
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.evaluate.evaluate import Evaluator
import os

from steer.utils.steer_eval_utils import  color, process_orig, evaluate_steer_eval, load_steer_eval_datasets, get_prompt

MODEL ='DeepSeek-V3.2'

@hydra.main(version_base='1.2', config_path='./hparams/Steer/experiment_hparams/steer_eval', config_name='steer_eval.yaml')
def main(top_cfg: DictConfig):
    print("Global Config:", top_cfg, "\n")
    
    # Initialize and load
    loader = DatasetLoader()
    train_datasets = loader.load_file(dataset_name=top_cfg.dataset,split='train')
    eval_datasets = loader.load_file(dataset_name=top_cfg.dataset,split='generation')
    eval_datasets = eval_datasets['valid'] if top_cfg.exp == 'valid' else eval_datasets['eval']
    
    all_generation_results = []
    
    tmp_generation_file_path = f"{top_cfg.generation_output_dir}/tmp_generation_results_{top_cfg.exp}.json"
    os.makedirs(os.path.dirname(tmp_generation_file_path), exist_ok=True)
    continue_generation_results = top_cfg.continue_generation_results if 'continue_generation_results' in top_cfg else True
    
    if  continue_generation_results and os.path.exists(tmp_generation_file_path):
        print(f"Loading existing generation results from {tmp_generation_file_path}...")
        with open(tmp_generation_file_path, "r", encoding="utf-8") as f:
            all_generation_results = json.load(f)
        for result in all_generation_results:
            print(f"{color.RED}Loaded results for concept {result['concept_id']}{color.END}")
            if result["concept_id"] in eval_datasets and result["concept_id"] in train_datasets:
                print(f"{color.BLUE}remove data from train/eval datasets: {result['concept_id']}{color.END}")
                train_datasets.pop(result["concept_id"])
                eval_datasets.pop(result["concept_id"])
        print(f"{color.RED}Continuing generation for remaining {len(train_datasets)} concepts...{color.END}")


    generate_vector = top_cfg.generate_vector
    generate_response = top_cfg.generate_response
    need_evaluate = top_cfg.evaluate
    method = top_cfg.method

    if generate_vector and method != "prompt": 
        vector_generator = BaseVectorGenerator(top_cfg)
        print(f"{color.GREEN}Vector Generator Hparams:{color.END}\n", vector_generator.hparams_dict)
    if generate_response :
        vector_applier = BaseVectorApplier(top_cfg)
        print(f"{color.GREEN}Vector Applier Hparams:{color.END}\n", vector_applier.hparams_dict)
        if top_cfg.best_multip_path != 'None':
            print(f"{color.BLUE}Using best multipliers from: {top_cfg.best_multip_path}{color.END}")
            best_multipliers = json.loads(  open( top_cfg.best_multip_path, "r", encoding="utf-8").read()  )
            print(f"{color.BLUE}Best Multipliers:{color.END}\n", best_multipliers)
        else:
            best_multipliers = None
    vectors = None
    for i, concept_train_data in train_datasets.items():
        if generate_vector:
            if method == "prompt":
                print(f"{color.BLUE}Processing concept {i} with {len(concept_train_data)} items{color.END}")
                vector_path = f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/{method}/layer_{top_cfg.layers[0]}.json"
                if os.path.exists(vector_path) :
                    print(f"{color.RED}Vectors already exist at {vector_path}, skipping generation.{color.END}")
                else:
                    generated_prompt = get_prompt(concept_train_data, n_shots=-1)
                    print(f"Generated prompt for concept {i}: {generated_prompt}")
                    os.makedirs( os.path.dirname(vector_path) , exist_ok=True)
                    with open( vector_path , "w", encoding="utf-8") as f:
                        json.dump( { 'prompt': generated_prompt } , f, indent=2, ensure_ascii=False)
            else:
                print(f"{color.BLUE}Processing concept {i} with {len(concept_train_data)} items{color.END}")
                train_dataset_for_concept = {
                    f'steer_eval_concept_{i}': concept_train_data
                }
                vector_path = f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/{method}/layer_{top_cfg.layers[0]}.pt"
                if os.path.exists(vector_path) :
                    print(f"{color.RED}Vectors already exist at {vector_path}, skipping generation.{color.END}")
                else:
                    vectors = vector_generator.generate_vectors(train_dataset_for_concept)[f'steer_eval_concept_{i}']
                    print(f"Generated vectors for concept {i}: {vectors}")
                
        if generate_response:
            print(f"{color.BLUE}Generating responses for concept {i}...{color.END}")
            
            if method == "prompt":
                with open(f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/prompt/layer_{top_cfg.layers[0]}.json", "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                n_shots = top_cfg.multipliers[0]
                print(f"{color.BLUE}Processing concept {i} generating prompt with {n_shots} shots...{color.END}")
                vector_applier.hparams_dict['prompt'].prompt = get_prompt(concept_train_data, 
                                                                          n_shots=n_shots,
                                                                          orig_prompt = prompt_data['prompt'])
                print(f"{color.YELLOW}{vector_applier.hparams_dict['prompt'].prompt}{color.END}")
                vector_applier.apply_vectors()  
            elif generate_vector and vectors is not None:
                vector_applier.apply_vectors(vectors)  
            else:
                if method == "caa":
                    if top_cfg.use_pca:
                        vector_applier.hparams_dict[method].steer_vector_load_dir = f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/pca_vector"
                    else:
                        vector_applier.hparams_dict[method].steer_vector_load_dir = f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/caa_vector"
                elif method == "reps_vector":
                    vector_applier.hparams_dict[method].steer_vector_load_dir = f"{top_cfg.steer_vector_output_dirs[0]}/steer_eval_concept_{i}/reps_vector"
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if best_multipliers is not None:
                    vector_applier.hparams_dict[method].multipliers=[ int(best_multipliers[i]) ] 
                print(f"{color.BLUE}{vector_applier.hparams_dict[method]}{color.END}") 
                vector_applier.apply_vectors()
            eval_data = eval_datasets[i]
            # Generate results using the vector applier
            generated_results = vector_applier.generate(
            {
                'steer_eval_eval': eval_data
            },
            save_results=False
            )
            
            save_data = {
                'concept_id': i,
                'concept_name': concept_train_data[0]['concept'],
                'llm_description': concept_train_data[0]['llm_description'],
                'generation_prompt': vector_applier.model.prompt if hasattr(vector_applier.model, 'prompt') else None,
                'generated_results': generated_results
            }
            all_generation_results.append(save_data)
            
            vector_applier.model.reset_all()
            
            with open(tmp_generation_file_path, "w", encoding="utf-8") as f:
                json.dump(all_generation_results, f, indent=2, ensure_ascii=False)
    if generate_response:                
        print("Saving all generation results to file...")
        generation_file_path = f"{top_cfg.generation_output_dir}/all_generation_results_{top_cfg.exp}.json"
        os.makedirs(os.path.dirname(generation_file_path), exist_ok=True)
        with open(generation_file_path, "w", encoding="utf-8") as f:
            json.dump(all_generation_results, f, indent=2, ensure_ascii=False)
    
        print(f"All generate results saved to files: {generation_file_path}")
    
    #########################################################

    if need_evaluate :
        evaluate_steer_eval(  MODEL , top_cfg.generation_output_dir)

    if top_cfg.generate_orig_output:
        process_orig(top_cfg.generation_output_dir)
        base_save_path = top_cfg.generation_output_dir.rsplit('/', 1)[0] + '/base'
        if need_evaluate :
            evaluate_steer_eval( MODEL , base_save_path)


if __name__ == '__main__':
    main()
