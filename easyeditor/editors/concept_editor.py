import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from ..util.globals import *
from ..evaluate import compute_concept_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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


# class ConceptEditor(BaseEditor):
class ConceptEditor:

    @classmethod
    def from_hparams(cls, hparams: HyperParams, prompt_hparams: Dict= None):
        if hparams is None :
            if prompt_hparams is None:
                raise NotImplementedError
            phparams = HyperParams()
            phparams.alg_name = 'prompt'
            phparams.model_name = prompt_hparams['model_name']
            phparams.device = prompt_hparams['device']
            phparams.max_length = 40
            phparams.model_parallel = False
            return cls(phparams)
        return cls(hparams)
    
    # def __init__(self):
    #     super().__init__()

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        if hparams.alg_name != 'prompt':
            self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            # if 't5' in self.model_name.lower():
            #     self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = T5Tokenizer.from_pretrained(self.model_name)
            # elif 'gpt-3.5' in self.model_name.lower():
            #     self.model, self.tok = None, None
            if 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'baichuan' in self.model_name.lower():
            #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'chatglm' in self.model_name.lower():
            #     self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.unk_token_id = 64787
            #     # self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'internlm' in self.model_name.lower():
            #     self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            #     self.tok.pad_token_id = self.tok.eos_token_id
            # elif 'qwen' in self.model_name.lower():
            #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams


    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             instance_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        concept_consistency = kwargs['concept_consistency'] if 'concept_consistency' in kwargs.keys() else False
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
            requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                            locality_inputs, instance_inputs, **kwargs)
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')
        
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                metrics = {
                    "pre": compute_concept_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, test_concept_consistency=False)
                }
                all_metrics.append(metrics)
        for i, request in enumerate(requests):
            start = time()

            if self.alg_name == 'prompt':
                PMT = f"Definition of {request['subject']}: {request['target_new']}\n"
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_concept_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, test_concept_consistency=concept_consistency, P=PMT),
                })
                
                edited_model = self.model
                weights_copy = None
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds= None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_concept_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_concept_consistency=concept_consistency),
                })
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                            len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                    locality_result = []
                    for ans,label in zip(all_metrics[i]['post']['locality'][f'{locality_key}_output'],all_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                        locality_result.append(np.mean(np.equal(ans, label)))
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                all_metrics[i]['pre'].pop('locality')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
        
        return all_metrics, edited_model, weights_copy

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          instance_inputs: Optional[Dict] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'instance': {},
            'locality': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
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

        if instance_inputs is not None:
            for instance_key in instance_inputs.keys():
                if isinstance(instance_inputs[instance_key]['prompt'], str):
                    instance_inputs[instance_key]['prompt'] = [instance_inputs[instance_key]['prompt'],]
                for i, request in enumerate(requests):
                    if instance_inputs[instance_key]['prompt'][i] is not None:
                        request['instance'].update(
                            {
                                instance_key: {
                                    'prompt': instance_inputs[instance_key]['prompt'][i]
                                }
                            }
                        )
        return requests

    def b(self):
        print("ConceptEditor's b function")