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
        # for loreft generation
        self.reft_model = None
        
    def _load_model(self):
        if self.model is None:
            from ..models import get_model
            self.model, self.tokenizer = get_model(self.config)
            self.device = self.model.device
    def reset_loreft(self):
        if self.hparams_dict.get('loreft') is not None:
            self.reft_model = None
    
    def apply_steering(self, hparams_dict, model=None, vectors=None):
        from ..utils.alg_dict import METHODS_CLASS_DICT  
        for alg_name in hparams_dict.keys():
            if alg_name in METHODS_CLASS_DICT:
                set_seed(hparams_dict[alg_name].seed)
                # print(f"Applying {alg_name} vectors to model ...")
                if alg_name == 'prompt':
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name] , model)
                elif alg_name == "loreft":
                    reft_model, model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name])
                    self.reft_model = reft_model
                elif vectors is None or vectors.get(alg_name) is None:
                    assert hparams_dict[alg_name].steer_vector_load_dir is not None, f"Steer vector load path {hparams_dict[alg_name].steer_vector_load_dir} does not exist !"
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name] , model)
                else:
                    model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name],  model, vectors[alg_name])
                print(f"Applying {alg_name} vectors or prompt to model successfully !\n")
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

    def generate(self, datasets, save_results=True, **kwargs):
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
            num_responses = self.config.get('num_responses', 1)
            # judge method type
            if self.hparams_dict.get('loreft') is not None:
                preds,orig_preds,complete_output = self.loreft_generate(dataset, num_responses)
            else:
                for item in tqdm(dataset, desc=f"Evaluating dataset {generation_data_name}"):
                    if not item.get('input'):
                        continue
                    current_preds = []
                    current_output = []
                    input_text = self._process_input_text(item['input'])
                    inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens = not self.config.use_chat_template).to(self.device)

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
    

    def loreft_generate(self,dataset,num_responses):
        import pyreft
        import transformers
        from  torch.utils.data import DataLoader
        import datasets
        from .loreft.apply_loreft_intervention import InterventionEvalDataCollator
        from ..datasets.loreft_data import load_reft_eval_data
        def make_eval_data_module(
                tokenizer:transformers.PreTrainedTokenizer,
                model,df,positions="all",
                num_interventions = 1,
                nonstop = True,
                share_weights = True,
                max_length = 512
        ):
            all_base_input_ids, all_intervention_locations = [], []
            for row in df:
                base_prompt = row["input"]
                base_prompt_ids = tokenizer(
                    base_prompt, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                base_prompt_length = len(base_prompt_ids)
                if positions == "all_prompt":
                    intervention_locations = torch.tensor([[i for i in range(base_prompt_length)]])
                else:
                    first_n, last_n = pyreft.parse_positions(positions)
                    intervention_locations = pyreft.get_intervention_locations(
                        last_position=base_prompt_length, 
                        first_n=first_n, 
                        last_n=last_n,
                        pad_mode="first",
                        num_interventions=num_interventions,
                        share_weights=share_weights,
                    )
                all_base_input_ids.append(base_prompt_ids)
                all_intervention_locations.append(intervention_locations)     
            eval_dataset = datasets.Dataset.from_dict({
                "input_ids": all_base_input_ids,
                "intervention_locations": all_intervention_locations,
            })
            eval_dataset.set_format(
                type='torch', columns=[
                    'input_ids', 'intervention_locations',])
            data_collator_fn = transformers.DefaultDataCollator(
                return_tensors="pt"
            )
            data_collator = InterventionEvalDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
            return dict(train_dataset=None, eval_dataset=eval_dataset, data_collator=data_collator)
        hparams = self.hparams_dict["loreft"]
        eval_dataset = load_reft_eval_data(dataset,None, self.tokenizer,"You are a helpful assistant.", use_chat_template=True)
        batch_size = 1 # it will be something wrong if it is not 1
        eval_output_length = hparams.max_length
        temperature = hparams.temperature if hasattr(hparams, "temperature") else 1.0
        reft_layers = hparams.reft_layers
        number_of_interventions = len(reft_layers)
        position = "l1" if not hasattr(hparams,"position") else hparams.position
        data_module = make_eval_data_module(
            tokenizer=self.tokenizer,
            model=self.model.model,
            df=eval_dataset,
            positions=position,
            num_interventions=number_of_interventions,
            nonstop=True,
            share_weights=True,
            max_length=hparams.max_length
        )
        eval_dataloader = DataLoader(
            data_module["eval_dataset"],shuffle=False,
            batch_size=batch_size,
            collate_fn = data_module["data_collator"],
        )
        self.reft_model.set_device("cuda")
        result_generations = []
        result_origins = []
        result_complete = []
        for j in range(num_responses):
            all_generations = []
            all_origins = []
            all_complete = []
            for i, bactch in enumerate(eval_dataloader):
                inputs = {k: v.to(self.device) for k, v in bactch.items()}
                if "intervention_locations" in inputs:
                    if inputs["intervention_locations"].dim() == 3:
                        unit_locations={"sources->base": (
                            None,
                            inputs["intervention_locations"].permute(1, 0, 2).tolist()
                        )}
                    else:
                        # this is dummy for lora only baseline
                        unit_locations={"sources->base": (None, 0)}
                origin_outputs, intervention_outputs = self.reft_model.generate(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, 
                    unit_locations=unit_locations, intervene_on_prompt=True, 
                    subspaces=[{"idx":[0]}] * number_of_interventions,
                    max_new_tokens=eval_output_length, do_sample=True, 
                    temperature=temperature,output_original_output = True
                )
                # Decode and print only the generated text without prompt tokens
                input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
                generated_texts = [
                    self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                    for generation, input_length in zip(intervention_outputs, input_lengths)
                ]
                origin_texts = [
                    self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                    for generation, input_length in zip(origin_outputs, input_lengths)
                ]
                complete_output = [
                    self.tokenizer.decode(generation, skip_special_tokens=False)
                    for generation in intervention_outputs
                ]
                all_generations += generated_texts
                all_origins += origin_texts
                all_complete += complete_output
            for idx in range(len(all_generations)):
                if j == 0:
                    result_generations.append([all_generations[idx]])
                    result_origins.append([all_origins[idx]])
                    result_complete.append([all_complete[idx]])
                else:
                    result_generations[idx].append(all_generations[idx])
                    result_origins[idx].append(all_origins[idx])
                    result_complete[idx].append(all_complete[idx])
        return result_generations, result_origins, result_complete