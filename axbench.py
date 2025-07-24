import json
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
    train_data = dataset_loader.load_file('axbench', 'train')
    
    # train_data is now a list of tuples: [(concept_id, concept_data_list), ...]
    concept_grouped_data = {}
    for concept_id, concept_data_list in train_data:
        concept_grouped_data[concept_id] = concept_data_list

    eval_data = dataset_loader.load_file('axbench', 'generation')  # The Apacha-Eval dataset
    return concept_grouped_data, eval_data

@hydra.main(version_base='1.2', config_path='./hparams/Steer/experiment_hparams/axbench', config_name='axbench.yaml')
def main(top_cfg: DictConfig):
    print("Global Config:", top_cfg, "\n")

    # Initialize and load
    train_datasets, eval_datasets = load_axbench_datasets()
    all_evaluation_results = []
    all_generation_results = []  
    
    vector_generator = BaseVectorGenerator(top_cfg)
    vector_applier = None
    
    eval_args = {"mode": 'direct', "save_results": False, "eval_methods": ["llm"], "llm_model": "gpt-4o-mini-2024-07-18" , 'save_path': top_cfg.generation_output_dir}
    evaluator = Evaluator(**eval_args)

    for i, concept_train_data in train_datasets.items():
        if i >= 2:
            continue
        print(f"Processing concept {i} with {len(concept_train_data)} items")
        train_dataset_for_concept = {
            f'axbench_concept_{i}': concept_train_data
        }
        vectors = vector_generator.generate_vectors(train_dataset_for_concept)[f'axbench_concept_{i}']
        print(f"Generated vectors for concept {i}: {vectors}")

        if vector_applier is None:
            vector_applier = BaseVectorApplier(top_cfg)
        
        if "prompt" in vector_applier.hparams_dict:
            vector_applier.hparams_dict['prompt'].prompt = concept_train_data[0]['output_concept']
        vector_applier.apply_vectors(vectors)   # Different from axbench, the steering factor is fixed for each concept
        
        # Randomly sample 10 items from apacha-eval
        sampled_eval_data = random.sample(eval_datasets, 10)
        
        # Generate results using the vector applier
        generated_results = vector_applier.generate(
           {
            'axbench_eval': sampled_eval_data
           },
           save_results=False
        )
        
        save_data = {
            'concept_id': i,
            'concept_name': concept_train_data[0]['output_concept'],
            'generated_results': generated_results,
            'eval_data': sampled_eval_data
        }
        all_generation_results.append(save_data)
        
        vector_applier.model.reset_all()

    print("Saving all generation results to file...")
    generation_file_path = f"{eval_args['save_path']}/all_generation_results.json"
    os.makedirs(os.path.dirname(generation_file_path), exist_ok=True)
    with open(generation_file_path, "w", encoding="utf-8") as f:
        json.dump(all_generation_results, f, indent=2, ensure_ascii=False)
    
    print("Evaluating all generation results...")
    all_average_scores = 0
    for concept_info in all_generation_results:
        concept_id = concept_info['concept_id']
        concept_name = concept_info['concept_name']
        generated_results = concept_info['generated_results']
        
        eval_results = evaluator.evaluate_from_direct(
            generated_results, 
            f"axbench_eval_{concept_id}", 
            concept=concept_name
        )
        all_average_scores += eval_results['mean_aggregated_rating']
        all_evaluation_results.append({
            f'axbench_concept_{concept_id}': eval_results
        })
        
        print(f"Evaluated concept {concept_id}: {concept_name}")
    
    
    evaluation_save_results={
        'all_average_scores': all_average_scores/len(all_generation_results),
        'all_evaluation_results': all_evaluation_results
    }
        
    evaluation_file_path = f"{eval_args['save_path']}/all_evaluation_results.json"
    os.makedirs(os.path.dirname(evaluation_file_path), exist_ok=True)
    with open(evaluation_file_path, "w", encoding="utf-8") as f: 
        json.dump(evaluation_save_results, f, indent=2, ensure_ascii=False)
    
    print(f"All results saved to files: {generation_file_path} and {evaluation_file_path}")

if __name__ == '__main__':
    main()

