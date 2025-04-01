import json
import os
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from ..utils.seed import set_seed

class BaseVectorApplier:
    """Base vector applier for all methods"""
    
    def __init__(self, top_cfg: DictConfig):
        from ..utils import load_apply_vector_hparams
        self.hparams_dict = load_apply_vector_hparams(top_cfg)
        for alg_name, hparams in self.hparams_dict.items():
            print(f"{alg_name.upper()} Applier Hyperparameters:\n{hparams}")

        self.config = top_cfg
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        if self.model is None:
            from ..models import get_model
            self.model, self.tokenizer = get_model(self.config)
            self.device = self.model.device
    
    def apply_steering(self, hparams_dict, model=None, vectors=None):
        from ..utils.alg_dict import METHODS_CLASS_DICT  
        for alg_name in hparams_dict.keys():
            if alg_name in METHODS_CLASS_DICT:
                set_seed(hparams_dict[alg_name].seed)
                # print(f"Applying {alg_name} vectors to model ...")
                if vectors is None or  vectors.get(alg_name) is None:
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name] , model)
                else:
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name],  model, vectors[alg_name])
                print(f"Applying {alg_name} vectors to model successfully !\n")
            else:
                return NotImplementedError(f"Method {alg_name} not implemented !")
 
        return model

    def apply_vectors(self, vectors=None):
        if self.hparams_dict:
            self.model = self.apply_steering(self.hparams_dict, self.model, vectors)
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
        if self.model is None:
            self._load_model()
        self.model.model.eval()

    def _process_input_text(self, input_text: str) -> str:
        if hasattr(self.model, 'prompt'):
            base_prompt = self.model.prompt
            input_text = f"{base_prompt} {input_text}"
            
        from steer.utils.templates import build_model_input
        input_text = build_model_input(input_text, self.tokenizer, self.config.system_prompt, self.config.use_chat_template)
            
        return input_text

    def generate(self, datasets, **kwargs):
        # Use kwargs parameters if provided, otherwise use parameters from config
        generation_params = dict(kwargs) if kwargs else dict(self.config.generation_params)
        generation_data_names = list(datasets.keys())
        # Set pad_token_id to eos_token_id if not already set in generation_params
        if 'pad_token_id' not in generation_params:
            generation_params['pad_token_id'] = self.tokenizer.eos_token_id
        for generation_data_name in generation_data_names:

            save_file_path = os.path.join(self.config.generation_output_dir, f"{generation_data_name}_results.json")
            if os.path.exists(save_file_path):  
                print(f"\033[1;34mFile {save_file_path} already exists! The result will be overwritten!\033[0m")

            orig_preds = []

            preds = []
            complete_output=[]
            generation_data_size = self.config['generation_data_size']
            if generation_data_size is None:
                generation_data_size = -1
            dataset = datasets[generation_data_name][:generation_data_size] if generation_data_size > 0 else datasets[generation_data_name]
                
            for item in tqdm(dataset, desc=f"Evaluating dataset {generation_data_name}"):
                if not item.get('input'):
                    continue
                current_preds = []
                current_output = []
                input_text = self._process_input_text(item['input'])
                inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens = not self.config.use_chat_template).to(self.device)
                
                num_responses = self.config.get('num_responses', 1)
                for j in range(num_responses):
                    if num_responses > 1:
                        set_seed(j)
                    with torch.no_grad():
                        if self.config.get('steer_from_end_position', False):
                            instr_pos = self.find_instruction_end_postion(inputs['input_ids'][0])
                            print("Steering from end position:", instr_pos)
                            self.model.set_from_positions(instr_pos)  
                        output = self.model.model.generate(**inputs, **generation_params)
                        current_output.append(self.tokenizer.decode(output[0], skip_special_tokens=False))
                        output=output[0][inputs['input_ids'].shape[1]:]
                        text = self.tokenizer.decode(output, skip_special_tokens=True)
                        current_preds.append(text)
                preds.append(current_preds)
                complete_output.append(current_output)

                if self.config.get('generate_orig_output', False):
                    output = self.model.ori_generate(**inputs, **generation_params)
                    output=output[0][inputs['input_ids'].shape[1]:]
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    orig_preds.append([text])

            formatted_results = self._format_result(dataset, orig_preds=orig_preds,preds=preds, complete_output=complete_output)
            self.save_results(formatted_results, generation_data_name)
        item = formatted_results[0]
        print(f"\n===== {generation_data_name} Results =====\n")
        print(f"----- Input -----\n{item['input']}\n")
        if self.config.get('generate_orig_output', False):
            print(f"----- Orig Output-----\n{item['orig_pred']}\n")
        print(f"----- Steered Output-----\n{item['pred']}\n")

        return formatted_results
    
    
    def save_results(self, results, dataset_name):
        os.makedirs(self.config.generation_output_dir, exist_ok=True)
        
        output_file = os.path.join(
            self.config.generation_output_dir, 
            f"{dataset_name}_results.json"
        )
        print(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                results, 
                f,
                indent=4,
                ensure_ascii=False
            )

    def _format_result(self, dataset, orig_preds,preds, complete_output):
        results = []
        for idx in range(len(preds)):
            item = dataset[idx] 
            result = {
                "input": item.get("input"),
                "orig_pred": orig_preds[idx] if self.config.get('generate_orig_output', False) else [],
                "pred": preds[idx],
                "reference_response": item.get("reference_response"),
                'complete_output': complete_output[idx]
            }
            results.append(result)
        return results

    def find_instruction_end_postion(self, tokens):
        start_pos = tokens.size(0) - 1
        return start_pos
    
