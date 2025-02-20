from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from ..dataset.processor.llavaov_processors import LLaVAOneVisionProcessor
from ..dataset.processor.qwen2vl_processors import Qwen2VLProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import math
import random
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration

from ..util.globals import *
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_hf_edit_results)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor:
    """Multimodal editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT
                
                self.model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint
                ) 
                # Get tokenizer and vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
                self.vis_tok = vis_processor
                if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                    tok_name = (
                        hparams.tokenizer_name
                        if hparams.tokenizer_name is not None
                        else hparams.name
                    )
                    tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                        tok_name
                    )            
                    if tokenizer.pad_token == None or tokenizer.pad_token == '':
                        tokenizer.pad_token = tokenizer.eos_token    
                    self.tok = tokenizer
                    
                self.vis_root = hparams.coco_image
                self.rephrase_root = hparams.rephrase_image     
                        
            elif hparams.model_name == "minigpt4":
                from ..trainer.blip2_models import MiniGPT4
                
                self.model = MiniGPT4(
                    vit_model="eva_clip_g",
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    llama_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                )   
                # Get tokenizer and vis_processor
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
                self.vis_tok = vis_processor
                if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                    tok_name = (
                        hparams.tokenizer_name
                        if hparams.tokenizer_name is not None
                        else hparams.name
                    )
                    tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                        tok_name
                    )            
                    if tokenizer.pad_token == None or tokenizer.pad_token == '':
                        tokenizer.pad_token = tokenizer.eos_token    
                    self.tok = tokenizer     
                
                self.vis_root = hparams.coco_image
                self.rephrase_root = hparams.rephrase_image  
                
            elif "llava-onevision" in hparams.model_name.lower():   
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    hparams.model_name,
                    torch_dtype=torch.float32,
                    # attn_implementation="flash_attention_2" 
                )
                self.vis_tok = LLaVAOneVisionProcessor()
                self.tok = AutoProcessor.from_pretrained(hparams.model_name)
                self.model_name = "llava-onevision"

            elif "qwen2-vl" in hparams.model_name.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    hparams.model_name, 
                    torch_dtype=torch.float32,
                    # attn_implementation="flash_attention_2"
                )
                self.vis_tok = Qwen2VLProcessor()
                self.tok = AutoProcessor.from_pretrained(hparams.model_name)
                self.model_name = "qwen2-vl"
                
        else:
            self.model, self.tok = self.model_name
            
        self.model.to(f'cuda:{hparams.device}')
        self.hparams = hparams

    def edit(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            file_type: Union[str,List[str]],
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            keep_original_weight=False,
            verbose=True,
            sequential_edit=False,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        # assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs, file_type,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
            assert self.hparams.batch_size == 1, f'Single Edit, pls set the batch_size to 1....'

        # sequential editing
        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.alg_name == 'IKE' and self.hparams.k == 0:
                    edited_model = self.model
                    weights_copy = None
                else:
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=None
                )
            exec_time = time() - start
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
                
            all_metrics = []
            exec_time = time() - start
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                if self.alg_name == 'IKE':
                    if self.hparams.k != 0:    
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                                request, self.hparams.device),
                            "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                                request, self.hparams.device, pre_edit=True)
                        }
                    else:
                        # QUESTION = request["prompt"]
                        # ANSWER = request["target"]
                        from copy import deepcopy
                        prompt_new_request = deepcopy(request)
                        prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
                        prompt_new_request["prompt"] = prefix_template + request["prompt"]
                        prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
                        prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
                        prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                prompt_new_request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device)
                        }
                else:
                    if self.model_name in ['minigpt4', 'blip2']:
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device)
                        }
                    elif self.model_name in ['llava-onevision', 'qwen2-vl']:
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device)
                        }
                                
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
                    post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
                    metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
                    post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')

                LOG.info(f"Evaluation took {time() - start}")
                all_metrics.append(metrics)
        
        # single editing
        else:
            all_metrics = []
            for i, request in tqdm(enumerate(requests),total=len(requests)):
                start = time()
                if self.alg_name == 'IKE':
                    if self.hparams.k != 0:                    
                        assert 'train_ds' in kwargs.keys(), 'IKE need train_ds (For getting In-Context prompt)'
                        edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                            self.model,
                            self.tok,
                            request,
                            self.hparams,
                            copy=False,
                            return_orig_weights=True,
                            keep_original_weight=keep_original_weight,
                            train_ds=kwargs['train_ds']
                        )
                    else:
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
                        keep_original_weight=keep_original_weight
                    )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                if self.alg_name == 'IKE':
                    if self.hparams.k != 0:
                        metrics = {
                            'case_id': i,
                            "time": exec_time,
                            "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                                request, self.hparams.device),
                            "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                                request, self.hparams.device, pre_edit=True)
                        }
                    else:
                        from copy import deepcopy
                        prompt_new_request = deepcopy(request)
                        prefix_template = self.hparams.template.format(prompt=request["prompt"],target=request["target"])
                        prompt_new_request["prompt"] = prefix_template + request["prompt"]
                        prompt_new_request["rephrase_prompt"] = prefix_template + request["rephrase_prompt"]
                        prompt_new_request["locality_prompt"] = prefix_template + request["locality_prompt"]
                        prompt_new_request["multimodal_locality_prompt"] = prefix_template + request["multimodal_locality_prompt"]
                        metrics = {
                            'case_nums': i,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                prompt_new_request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device)
                
                        }
                else:
                    if self.model_name in ['minigpt4', 'blip2']:
                        metrics = {
                            'case_id': i,
                            "time": exec_time,
                            "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device)
                        }
                    elif self.model_name in ['llava-onevision', 'qwen2-vl']:
                        metrics = {
                            'case_id': i,
                            # "requested_rewrite": request,
                            "time": exec_time,
                            "post": compute_multimodal_hf_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device),
                            "pre": compute_multimodal_hf_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                                request, self.hparams.device),
                        }   
                     
                    
                if 'locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['locality_output']) == \
                            len(metrics['pre']['locality_output'])
                    base_logits = torch.tensor(metrics['pre']['locality_output']).to(torch.float32)
                    post_logits = torch.tensor(metrics['post']['locality_output']).to(torch.float32)
                    metrics['post']['locality_acc'] = sum(post_logits.view(-1) == base_logits.view(-1))/base_logits.view(-1).shape[0]
                    metrics['post'].pop('locality_output')
                    metrics['pre'].pop('locality_output')
                    
                if 'multimodal_locality_output' in metrics['post'].keys():
                    assert len(metrics['post']['multimodal_locality_output']) == \
                            len(metrics['pre']['multimodal_locality_output'])
                    base_image_logits = torch.tensor(metrics['pre']['multimodal_locality_output']).to(torch.float32)
                    post_image_logits = torch.tensor(metrics['post']['multimodal_locality_output']).to(torch.float32)
                    metrics['post']['multimodal_locality_acc'] = sum(post_image_logits.view(-1) == base_image_logits.view(-1))/post_image_logits.view(-1).shape[0]
                    metrics['post'].pop('multimodal_locality_output')
                    metrics['pre'].pop('multimodal_locality_output')


                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                    )

                all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0, \
        f'DataSet {ds} not supported yet.'

    #    assert self.alg_name == 'IKE', 'Only IKE supported for MultimodalEditor'
        num_edits = 1
        # num_edits = self.hparams.batch_size
        
        all_metrics = []

        for i, request in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):

            start = time()
            
            if 'template' in kwargs:
                request['prompt'] = kwargs['template'].format(request['prompt'])

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight
                )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            if self.alg_name == 'IKE':
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                        request, self.hparams.device),
                    "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                        request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    'case_id': i,
                    # "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok,
                                                        request, self.hparams.device),
                    "pre": compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok,
                                                        request, self.hparams.device)
                }
            if 'locality_output' in metrics['post'].keys():
                assert len(metrics['post']['locality_output']) == \
                        len(metrics['pre']['locality_output'])
                base_logits = metrics['pre']['locality_output'].to(torch.float32)
                post_logits = metrics['post']['locality_output'].to(torch.float32)
                if post_logits.shape[1] > base_logits.shape[1]:
                    post_logits = post_logits[:, -base_logits.shape[1]:, :]
                else:
                    base_logits = base_logits[:, -post_logits.shape[1]:, :]

                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=1, dim=-1).indices
                metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('locality_output')
                metrics['pre'].pop('locality_output')
                
            if 'multimodal_locality_output' in metrics['post'].keys():
                assert len(metrics['post']['multimodal_locality_output']) == \
                        len(metrics['pre']['multimodal_locality_output'])
                base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                if post_image_logits.shape[1] > base_image_logits.shape[1]:
                    post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                else:
                    base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('multimodal_locality_output')
                metrics['pre'].pop('multimodal_locality_output')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                )

                all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          file_type: Union[str, List[str]] = None,
                          **kwargs
                          ):
        if isinstance(file_type, List):
            assert len(file_type) == len(image) == len(prompts) == len(targets)
        
        if isinstance(image, str):
            image = [image, ]
        # image_path = [os.path.join(self.vis_root, image_) for image_ in image]
        # image = [Image.open(ip).convert("RGB") for ip in image_path]
        # image = [self.vis_tok(i).to(self.hparams.device) for i in image]
        image = [self.vis_tok(file=m, file_type=n) for m,n in zip(image, file_type)]
    
        
        requests = [{
            'prompt': prompt,
            'target': target,
            'image': image_,
            'file_type': file_type_
        }        
        for prompt, target, image_, file_type_ in zip(prompts, targets, image, file_type)
        ]
        
        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_inputs['text']['prompt']) == len(locality_inputs['text']['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(locality_inputs['vision']['prompt']) == len(locality_inputs['vision']['ground_truth']) \
                == len(locality_inputs['vision']['image']) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            # rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            # rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            rephrase_image = [self.vis_tok(file=m, file_type=n) for m, n in zip(rephrase_image,file_type)]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            # locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
            locality_image_path = [multimodal_locality_image_ for multimodal_locality_image_ in multimodal_locality_image]
            # locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
            locality_image = [self.vis_tok(file=i, file_type="image") for i in locality_image_path]
             
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
        
        if 'loc_prompts' in kwargs:
            if isinstance(kwargs['loc_prompts'], str):
                kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
            if len(kwargs['loc_prompts']) < len(requests):
                kwargs['loc_prompts'] = (kwargs['loc_prompts'] * math.ceil(len(requests) / len(kwargs['loc_prompts'])))[:len(requests)]
                random.shuffle(kwargs['loc_prompts'])
            assert len(kwargs['loc_prompts']) == len(prompts)

            for i, request in enumerate(requests):
                request.update(
                    {
                        'loc_prompt': kwargs['loc_prompts'][i]
                    }
                )

        return requests