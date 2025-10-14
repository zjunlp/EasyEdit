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
    def _multimodal_load_model(self):
        if self.model is None:
            from ..models import Multimodal_get_model
            self.model, self.tokenizer, self.processor = Multimodal_get_model(self.config)
            self.device = self.model.device
    def apply_steering(self, hparams_dict, model=None, vectors=None):
        from ..utils.alg_dict import METHODS_CLASS_DICT
        for alg_name in hparams_dict.keys():
            if alg_name in METHODS_CLASS_DICT:
                set_seed(hparams_dict[alg_name].seed)
                # print(f"Applying {alg_name} vectors to model ...")
                if alg_name == 'prompt':
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name] , model)
                elif vectors is None or vectors.get(alg_name) is None:
                    assert hparams_dict[alg_name].steer_vector_load_dir is not None, f"Steer vector load path {hparams_dict[alg_name].steer_vector_load_dir} does not exist !"
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name] , model)
                else:
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name],  model, vectors[alg_name])
                print(f"Applying {alg_name} vectors or prompt to model successfully !\n")
            else:
                return NotImplementedError(f"Method {alg_name} not implemented !")\
        
        return model
    def apply_vectors(self, vectors=None):
        if self.hparams_dict:
            self.model= self.apply_steering(self.hparams_dict, self.model, vectors)
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
        if self.model is None:
            self._load_model()
        self.model.model.eval() 
    def multimodal_apply_vectors(self, vectors=None):
        if self.hparams_dict:
            self.model = self.apply_steering(self.hparams_dict, self.model, vectors)
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
            # For multimodal models, set processor
            if hasattr(self.model, 'processor'):
                self.processor = self.model.processor
        if self.model is None:
            self._multimodal_load_model()
        self.model.model.eval()

    def _process_input_text(self, input_text: str) -> str:
        if hasattr(self.model, 'prompt'):
            base_prompt = self.model.prompt
            input_text = f"{base_prompt} {input_text}"
            
        from steer.utils.templates import build_model_input
        input_text = build_model_input(input_text, self.tokenizer, self.config.system_prompt, self.config.use_chat_template)
            
        return input_text

    def _process_multimodal_input(self, item) -> dict:
        """
        Handles multimodal input (text + image)

        Args:
            item: A data item containing the input and possibly an image

        Returns:
            A dictionary of processed inputs, including input_ids, attention_mask, etc
        """
        input_text = item.get('input', '')
        image = item.get('image', None)
        
        # If there is a basic prompt word, add it to the input text
        if hasattr(self.model, 'prompt'):
            base_prompt = self.model.prompt
            input_text = f"{base_prompt} {input_text}"
        
        # Building multimodal input
        if image is not None:
            # With images: Building multimodal dialogues
            from steer.utils.templates import build_multimodal_model_input
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": input_text}, {"type": "image"}]}
            ]
            processed_text = build_multimodal_model_input(
                conversation, 
                self.processor, 
                self.config.system_prompt, 
                self.config.use_chat_template
            )
            
            # Using processors to handle multimodal input
            inputs = self.processor(
                text=processed_text,
                images=image,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)
        else:
            # Only text: use normal text processing
            from steer.utils.templates import build_multimodal_model_input
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": input_text}]}
            ]
            processed_text = build_multimodal_model_input(
                conversation, 
                self.processor, 
                self.config.system_prompt, 
                self.config.use_chat_template
            )
            
            # Using processors to handle multimodal input
            inputs = self.processor(
                text=processed_text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)
        
        return inputs

    def generate(self, datasets, save_results=True, **kwargs):
        # Use kwargs parameters if provided, otherwise use parameters from config
        generation_params = dict(kwargs) if kwargs else dict(self.config.generation_params)
        generation_data_names = list(datasets.keys())
        # Set pad_token_id to eos_token_id if not already set in generation_params
        for generation_data_name in generation_data_names:
            os.makedirs(self.config.generation_output_dir, exist_ok=True)
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
            
            if self.model.VLLM_model is not None:
                from vllm import SamplingParams
                vllm_sampling_params = SamplingParams(
                    temperature=generation_params.get("temperature", 1.0),
                    top_p=generation_params.get("top_p", 1.0),
                    max_tokens= generation_params.get("max_tokens", generation_params.get("max_new_tokens", 100)) if "max_tokens" in generation_params else generation_params.get("max_new_tokens", 100),
                    top_k=generation_params.get("top_k", -1),
                    presence_penalty=generation_params.get("presence_penalty", 0.0),
                    frequency_penalty=generation_params.get("frequency_penalty", 0.0),
                )
                num_responses = self.config.get('num_responses', 1)
                for j in range(num_responses):
                    if num_responses > 1:
                        set_seed(j)
                        
                if self.config.get('steer_from_end_position', False):
                    print("[Warning] steer_from_end_position not supported in vLLM mode.")
                
                if 'pad_token_id' not in generation_params:
                    generation_params['pad_token_id'] = self.tokenizer.eos_token_id

                input_batch = []
                valid_indices = [] 
                
                for idx, item in enumerate(dataset):
                    if not item.get('input'):
                        continue
                    
                    input_text = self._process_input_text(item['input'])
                    for j in range(num_responses):
                        input_batch.append(input_text)
                        valid_indices.append(idx)
            
                outputs = self.model.VLLM_model.generate(
                    prompts= input_batch,
                    sampling_params = vllm_sampling_params
                )

                current_item_outputs = {}  # {item_idx: [responses]}
                
                for i, output in enumerate(outputs):
                    item_idx = valid_indices[i]
                    text = output.outputs[0].text.strip()
                    
                    if item_idx not in current_item_outputs:
                        current_item_outputs[item_idx] = []
                    current_item_outputs[item_idx].append(text)

                for idx, item in enumerate(dataset):
                    if not item.get('input'):
                        continue
                    
                    if idx in current_item_outputs:
                        current_output = current_item_outputs[idx]
                        current_preds = current_output.copy()
                    else:
                        current_output = []
                        current_preds = []
                    
                    preds.append(current_preds)
                    complete_output.append(current_output)
            else:
                if 'pad_token_id' not in generation_params:
                    generation_params['pad_token_id'] = self.tokenizer.eos_token_id
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
            if save_results:
                    self.save_results(formatted_results, generation_data_name)
            item = formatted_results[0]

            print(f"\n===== {generation_data_name} Results =====\n")
            print(f"----- Input -----\n{item['input']}\n")
            if self.config.get('generate_orig_output', False):
                print(f"----- Orig Output-----\n{item['orig_pred']}\n")
            print(f"----- Steered Output-----\n{item['pred']}\n")

        return formatted_results

    def multimodal_generate(self, datasets, save_results=True, **kwargs):
        # Use kwargs parameters if provided, otherwise use parameters from config
        generation_params = dict(kwargs) if kwargs else dict(self.config.generation_params)
        generation_data_names = list(datasets.keys())
        # Set pad_token_id to eos_token_id if not already set in generation_params
        if 'pad_token_id' not in generation_params:
            generation_params['pad_token_id'] = self.processor.tokenizer.eos_token_id
        for generation_data_name in generation_data_names:
            os.makedirs(self.config.generation_output_dir, exist_ok=True)
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
                
                # Processing multimodal input
                inputs = self._process_multimodal_input(item)
                
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
                        current_output.append(self.processor.decode(output[0], skip_special_tokens=False))
                        output=output[0][inputs['input_ids'].shape[1]:]
                        text = self.processor.decode(output, skip_special_tokens=True)
                        current_preds.append(text)
                preds.append(current_preds)
                complete_output.append(current_output)

                if self.config.get('generate_orig_output', False):
                    output = self.model.ori_generate(**inputs, **generation_params)
                    output=output[0][inputs['input_ids'].shape[1]:]
                    text = self.processor.decode(output, skip_special_tokens=True)
                    orig_preds.append([text])

            formatted_results = self._format_result(dataset, orig_preds=orig_preds,preds=preds, complete_output=complete_output)
            if save_results:
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
    