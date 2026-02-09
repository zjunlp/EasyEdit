import json, os
import random
import sys

from openai import OpenAI
from steer.evaluate.evaluate import Evaluator
import re
MODEL ='gpt-5-mini'
BASE_URL="xxx"
API_KEY="xxx"

# T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language model in producing responses. 
# Objective: 
# Direct the model to include content followed to "%s" (the concept) in its responses. 
# Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
   
# Return only the final prompt without any additional text."""

T_GENERATE_STEERING_PROMPT = """Generate a brief prompt that instructs a language model to naturally incorporate the concept "%s" into its responses.

Requirements:
- The concept should be semantically integrated, not just appended or copied verbatim
- The incorporation should feel organic, even if tangentially related to the main query
- Avoid instructing the model to simply repeat the concept phrase

Return only the final prompt without any additional text."""

def save_json_compact_lists(data, filename, indent=2):
    json_str = json.dumps(data, ensure_ascii=False, indent=indent)
    json_str = re.sub(r'\[\s+', '[', json_str)  
    json_str = re.sub(r'\s+\]', ']', json_str)  
    json_str = re.sub(r',\s+', ', ', json_str)  

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    print(f"saved to {filename}")

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def load_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_steer_eval_datasets( datasets:str,  path:str) -> list:
    
    N_Train = 68
    N_Eval = 30
    N_Valid = 5
    # print(f"Loading 500 concepts from steer_eval...")
    if not path:
        raise ValueError("Path must be provided for loading steer_eval datasets.")
    train_path = f"./data/hc_steer_bench/data/{datasets}/result/{path}/train_results.json"
    eval_path = f"./data/hc_steer_bench/data/{datasets}/result/{path}/test_results.json"
    print(f"{color.GREEN}Loading train data from:\n - {train_path}{color.END}")
    print(f"{color.GREEN}Loading test data from:\n - {eval_path}{color.END}")

    train_data = load_file(train_path)
    eval_data = load_file(eval_path)
    valid_data = {}
    for concept, datas in train_data.items():
        if len(datas) > N_Train:
            train_data[concept] = datas[:N_Train]
        elif len(datas) % 2 == 1:
            train_data[concept] = datas[:-1]  
        print(f"{color.GREEN}Concept {concept} has {len(train_data[concept])} training items after adjustment.{color.END}")

    for concept, datas in eval_data.items():
        for item in datas:
            item['input'] = item.pop('question')
        if len(datas) > N_Eval:
            eval_data[concept] = datas[:N_Eval]
        if len(datas) > N_Valid + N_Eval:
            valid_data[concept] = datas[N_Eval:N_Eval + N_Valid]
        else:
            valid_data[concept] = datas[-N_Valid:]
        print(f"{color.GREEN}Concept {concept} has {len(eval_data[concept])} evaluation items.{color.END}")
        print(f"{color.GREEN}Concept {concept} has {len(valid_data[concept])} validation items.{color.END}")
    return train_data, eval_data , valid_data


def process_orig(load_path):
    
    print("Processing original outputs for comparison...")
    # Load original generation results
    print("Loading all generation results from file...")
    generation_file_path = f"{load_path}/all_generation_results_test.json"
    with open(generation_file_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    for concept_info in raw_results:
        generated_results_list = concept_info['generated_results']
        for generated_results in generated_results_list:
            generated_results.pop("complete_output")
            generated_results.pop("pred")
            generated_results["pred"] = generated_results.pop("orig_pred")
    all_generation_results = raw_results
    
    base_save_path = load_path.rsplit('/', 1)[0] + '/base'

    print("Saving all generation results to file...")
    generation_file_path = f"{base_save_path}/all_generation_results.json"
    os.makedirs(os.path.dirname(generation_file_path), exist_ok=True)
    with open(generation_file_path, "w", encoding="utf-8") as f:
        json.dump(all_generation_results, f, indent=2, ensure_ascii=False)
    
    return all_generation_results
  

def evaluate_steer_eval( model , save_path, suffix=''):

    print(f"Loading all generation results from file {save_path}...")
    generation_file_path = f"{save_path}/all_generation_results.json"
    with open(generation_file_path, "r", encoding="utf-8") as f:
        all_generation_results = json.load(f)

    print("Evaluating all generation results...")
    all_evaluation_results = []
    all_evaluation_results_detial = []
    eval_args = {"mode": 'direct', "save_results": False, "eval_methods": ["llm"], "llm_model": f"{model}" , 'save_path': save_path}
    evaluator = Evaluator(**eval_args)

    all_average_scores = 0
    all_average_scores_list = []
    
    all_average_concept_scores = 0
    all_average_concept_scores_list = []
    all_average_instruction_scores = 0
    all_average_instruction_scores_list = []
    all_average_fluency_scores = 0
    all_average_fluency_scores_list = []
    for concept_info in all_generation_results:
        concept_id = concept_info['concept_id']
        concept_name = concept_info['concept_name']
        generated_results = concept_info['generated_results']
        
        eval_results = evaluator.evaluate_from_direct(
            generated_results, 
            f"steer_eval_eval_{concept_id}", 
            concept=concept_name
        )
        
        concept_mean = sum(eval_results['concept_scores']) / len(eval_results['concept_scores'])
        instruction_mean = sum(eval_results['instruction_scores']) / len(eval_results['instruction_scores'])
        fluency_mean = sum(eval_results['fluency_scores']) / len(eval_results['fluency_scores'])
        
        all_average_scores += eval_results['mean_aggregated_rating']
        all_average_concept_scores += concept_mean
        all_average_instruction_scores += instruction_mean
        all_average_fluency_scores += fluency_mean

        all_average_scores_list.append(eval_results['mean_aggregated_rating'])
        all_average_concept_scores_list.append(concept_mean)
        all_average_instruction_scores_list.append(instruction_mean)
        all_average_fluency_scores_list.append(fluency_mean)
        all_evaluation_results.append({
            f'steer_eval_concept_{concept_id}': eval_results
        })
        
        detial_result = {}
        for id,item in enumerate(eval_results['all_contents']):
            detial_result[f'item_{id}'] = {
                'concept': concept_name,
                'input':  generated_results[id]['input'],
                'pred':  generated_results[id]['pred'],
                'concept_eval': item['concept'],
                'instruction_eval': item['instruction'],
                'fluency_eval': item['fluency']
            }
        all_evaluation_results_detial.append({
            f'steer_eval_concept_{concept_id}': detial_result
        })
        eval_results.pop('all_contents')
        print(f"Evaluated concept {concept_id}: {concept_name}")
    
    
    evaluation_save_results={
        'MODEL': model,
        'all_average_scores': all_average_scores/len(all_generation_results),
        'all_average_scores_list': all_average_scores_list,
        'all_average_concept_scores': all_average_concept_scores/len(all_generation_results),
        'all_average_concept_scores_list': all_average_concept_scores_list,
        'all_average_instruction_scores': all_average_instruction_scores/len(all_generation_results),
        'all_average_instruction_scores_list': all_average_instruction_scores_list,
        'all_average_fluency_scores': all_average_fluency_scores/len(all_generation_results),
        'all_average_fluency_scores_list': all_average_fluency_scores_list,
        'all_evaluation_results': all_evaluation_results
    }

    if suffix:
        evaluation_file_path = f"{eval_args['save_path']}/all_evaluation_results_{model}_{suffix}.json"
        detial_file_path = f"{eval_args['save_path']}/all_evaluation_results_detial_{model}_{suffix}.json"
    else:
        evaluation_file_path = f"{eval_args['save_path']}/all_evaluation_results_{model}.json"
        detial_file_path = f"{eval_args['save_path']}/all_evaluation_results_detial_{model}.json"
        
    os.makedirs(os.path.dirname(evaluation_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(detial_file_path), exist_ok=True)
    with open(evaluation_file_path, "w", encoding="utf-8") as f: 
        json.dump(evaluation_save_results, f,  indent=2, ensure_ascii=False)
    with open(detial_file_path, "w", encoding="utf-8") as f: 
        json.dump(all_evaluation_results_detial, f,  indent=2, ensure_ascii=False)

    
    print(f"All evaluation results saved to files: {evaluation_file_path}")



def get_prompt(concept_train_data, n_shots=3, orig_prompt=None):
    """
    从 concept_train_data random sample three few-shot case and get Prompt。
    
    Args:
        concept_train_data: List[dict], 每个 dict 包含 question, matching, not_matching
        
    Returns:
        str:  Few-shot Prompt
    """
    if orig_prompt is None:
        concept = concept_train_data[0]['concept']
        client = OpenAI( base_url=BASE_URL,  api_key=API_KEY )
        completion = client.chat.completions.create(
            model=MODEL, 
            messages = [
                {"role": "user", "content": T_GENERATE_STEERING_PROMPT % concept},
            ]
        )
        prompt_intro = completion.choices[0].message.content.strip()
    else:
        prompt_intro = orig_prompt
    
    
    if n_shots > 0:
        if len(concept_train_data) < n_shots:
            selected_samples = concept_train_data
        else:
            selected_samples = random.sample(concept_train_data, n_shots)
        prompt = prompt_intro + "\n\nHere are some examples of the task:\n"
        for i, sample in enumerate(selected_samples, 1):
            question = sample.get('question', '')
            matching = sample.get('matching', '')       
            prompt += f"### Example {i}\n"
            prompt += f"Question: {question}\n"
            prompt += f"Response: {matching}\n\n"
        prompt += "### Now, please generate for the following:\n\n"
    elif n_shots == 0:
        prompt = prompt_intro + "\n\n### Now, please generate for the following:\n\n"
    else:
        prompt = prompt_intro 
    return prompt


if __name__ == "__main__":

    
    # model = sys.argv[1]  # e.g., "gpt-5-mini
    # save_path =  sys.argv[2]  # e.g., "generation/gemma-2-9b-it/reasoning/try/reps_vector"
    # suffix = sys.argv[3] if len(sys.argv) > 3 else ''  # e.g., "5m_mini"
    # evaluate_steer_eval( model , save_path, suffix)
    # python steer_eval_utils.py gpt-4.1-mini generation/gemma-2-9b-it/version1/personality/f_version_1/base 
    
    model = sys.argv[1]  # e.g., "gpt-5-mini
    domains = ['language_features', 'personality', 'reasoning_patterns', 'sentiment']
    methods = ['base', 'prompt', 'caa', 'reps_vector']
    for domain in domains:
        for method in methods:
            path = f"generation/gemma-2-9b-it/version1/{domain}/f_version_1/{method}"
            suffix = ''
            evaluate_steer_eval( model , path, suffix)
    # python steer_eval_utils.py gpt-4.1-mini 