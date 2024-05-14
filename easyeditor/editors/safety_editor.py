import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
# from accelerate import Accelerator
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import compute_safety_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)


# class SafetyEditor(BaseEditor)
class SafetyEditor:

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            if 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
            else:
                raise NotImplementedError
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams


    def _locate_toxic_layer(self, model, tokenizer, requests, **kwargs):
        if isinstance(tokenizer, LlamaTokenizer):
            tokenizer.padding_side = 'right'
        toxic_layer = []
        input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
        with torch.no_grad():
            outputs = model(**input)
        hidden_states = outputs.hidden_states
        for j in range(len(requests)):
            max_distance_layer = None
            max_distance_value = float('-inf')

            for layer_index in range(1, len(hidden_states)):
                euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

                if euclidean_distance.item() > max_distance_value:
                    max_distance_value = euclidean_distance.item()
                    max_distance_layer = layer_index
            toxic_layer.append(max_distance_layer-1)
        return toxic_layer

    def edit(self,
             prompts: Union[str, List[str]],
             prompts_with_systemPrompt: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             locality_inputs_with_systemPrompt:  Optional[Dict] = None,
             general_prompt: Optional[Union[str, List[str]]] = None,
             general_prompt_with_systemPrompt: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for general knowledge constrains
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, general_prompt, locality_inputs, **kwargs)
            requests_with_systemPrompt = self._prepare_requests(prompts_with_systemPrompt, target_new, ground_truth, general_prompt_with_systemPrompt, locality_inputs_with_systemPrompt, **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')


        if "NLPCC" in kwargs and kwargs['NLPCC']:
            for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
                start = time()
                if len(self.hparams.layers) == 0:
                    self.hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request,])
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request_with_systemPrompt],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                edited_model.save_pretrained(kwargs['ckpt_save_dir'])
                print(f"edited model is saved in {kwargs['ckpt_save_dir']}")
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
              

        else:
            all_metrics = []
            if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
                metrics = kwargs['pre_edit']
                all_metrics = metrics
            else:
                for i, request in enumerate(tqdm(requests)):
                    metrics = {
                        "pre": compute_safety_edit_quality(self.model, self.tok, request,
                                                self.hparams.device, max_output_tokens=self.hparams.max_output_length)
                    }
                    all_metrics.append(metrics)
                if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                    ### Store the pre_edit metric to refrain computing repeatedly
                    json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)
            for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
                start = time()
                if len(self.hparams.layers) == 0:
                    self.hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request,])
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request_with_systemPrompt],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': kwargs["case_id"],
                    "requested_rewrite": request,
                    "post": compute_safety_edit_quality(edited_model, self.tok, request_with_systemPrompt, self.hparams.device, max_output_tokens=self.hparams.max_output_length),
                    "time": exec_time,
                })
                
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )

            if isinstance(edited_model, LORA):
                edited_model=edited_model.model
            #for melo
            return all_metrics, edited_model, weights_copy

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          general_prompt: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if general_prompt is None:
            requests = [{
                'prompt': prompt,
                'target_new': target_new_,
                'ground_truth': ground_truth_,
                'locality': {}
            }
            for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
            ]
        
        else:

            requests = [{
                'prompt': prompt,
                'target_new': target_new_,
                'ground_truth': ground_truth_,
                'general_prompt': general_prompt_,
                'locality': {}
            }
            for prompt, ground_truth_, target_new_, general_prompt_ in zip(prompts, ground_truth, target_new, general_prompt)
            ]

        
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        
        return requests
