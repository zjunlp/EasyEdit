from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .singleton_editor import SingletonEditor
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo)
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
                
                model = Blip2OPT(
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
            elif hparams.model_name == "minigpt4":
                from ..trainer.blip2_models import MiniGPT4
                
                model = MiniGPT4(
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
            self.model = model
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
        else:
            self.model, self.tok = self.model_name
            
        self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image

    def edit(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            keep_original_weight=False,
            verbose=True,
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
        assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        all_metrics = []
        for i, request in enumerate(requests):
            start = time()

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
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            metrics = {
                'case_id': i,
                # "requested_rewrite": request,
                "time": exec_time,
                "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                    request, self.hparams.device),
                "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                    request, self.hparams.device, pre_edit=True)
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

                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
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
    
    # # edit_demo will return the logits after/before editing
    # def edit_demo(self,
    #         prompts: Union[str, List[str]],
    #         targets: Union[str, List[str]],
    #         image: Union[str, List[str]],
    #         rephrase_prompts: Optional[Union[str, List[str]]] = None,
    #         rephrase_image: Optional[Union[str, List[str]]] = None,
    #         locality_inputs: Optional[dict] = None,
    #         keep_original_weight=False,
    #         verbose=True,
    #         **kwargs
    #         ):
    #     """
    #     `prompts`: list or str
    #         the prompts to edit
    #     `targets`: str
    #         the expected outputs
    #     `image`: dict
    #         for multimodal
    #     """
    #     assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
    #     if isinstance(prompts, List):
    #         assert len(prompts) == len(targets) == len(image)
    #     else:
    #         prompts, targets, image = [prompts,], [targets,], [image,]

    #     if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
    #         self.hparams.batch_size = 1

    #     requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
    #                                       **kwargs)

    #     if hasattr(self.hparams, 'batch_size') :
    #            assert self.hparams.batch_size == 1 or \
    #                   print(f'Single Edit, pls set the batch_size to 1....')

    #     all_metrics = []
    #     for i, request in enumerate(requests):
    #         start = time()

    #         assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
    #         edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
    #             self.model,
    #             self.tok,
    #             request,
    #             self.hparams,
    #             copy=False,
    #             return_orig_weights=True,
    #             keep_original_weight=keep_original_weight,
    #             train_ds=kwargs['train_ds']
    #         )
    #         exec_time = time() - start
    #         LOG.info(f"Execution {i} editing took {exec_time}")
    #         start = time()
    #         metrics = {
    #             'case_id': i,
    #             # "requested_rewrite": request,
    #             "time": exec_time,
    #             "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
    #                                                 request, self.hparams.device),
    #             "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
    #                                                 request, self.hparams.device, pre_edit=True)
    #         }
    #         if 'locality_output' in metrics['post'].keys():
    #             assert len(metrics['post']['locality_output']) == \
    #                     len(metrics['pre']['locality_output'])
    #             base_logits = metrics['pre']['locality_output'].to(torch.float32)
    #             post_logits = metrics['post']['locality_output'].to(torch.float32)
    #             if post_logits.shape[1] > base_logits.shape[1]:
    #                 post_logits = post_logits[:, -base_logits.shape[1]:, :]
    #             else:
    #                 base_logits = base_logits[:, -post_logits.shape[1]:, :]

    #             base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=10, dim=-1).indices
    #             post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=10, dim=-1).indices
    #             metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
    #             metrics['post'].pop('locality_output')
    #             metrics['pre'].pop('locality_output')
                
    #         if 'multimodal_locality_output' in metrics['post'].keys():
    #             assert len(metrics['post']['multimodal_locality_output']) == \
    #                     len(metrics['pre']['multimodal_locality_output'])
    #             base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
    #             post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
    #             if post_image_logits.shape[1] > base_image_logits.shape[1]:
    #                 post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
    #             else:
    #                 base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

    #             base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
    #             post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
    #             metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
    #             metrics['post'].pop('multimodal_locality_output')
    #             metrics['pre'].pop('multimodal_locality_output')

    #         LOG.info(f"Evaluation took {time() - start}")

    #         if verbose:
    #             LOG.info(
    #                 f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
    #             )

    #         all_metrics.append(metrics)

    #     return all_metrics, edited_model, weights_copy, post_logits, pre_logits 

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
        or print(f'DataSet {ds} not supported yet.')

        assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        num_edits = 1
        # num_edits = self.hparams.batch_size
        
        all_metrics = []

        for i, request in enumerate(tqdm(ds), desc='Editing dataset', total=len(ds)):

            start = time()

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
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            metrics = {
                'case_id': i,
                "time": exec_time,
                "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                    request, self.hparams.device),
                "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                    request, self.hparams.device, pre_edit=True)
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
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) for image_ in image]
        image = [Image.open(ip).convert("RGB") for ip in image_path]
        image = [self.vis_tok(i).to(self.hparams.device) for i in image]
        
        requests = [{
            'prompt': prompt,
            'target': target,
            'image': image_,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
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
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            rephrase_image = [self.vis_tok(i).to(self.hparams.device) for i in rephrase_image]
            
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
            
            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
            locality_image = [self.vis_tok(i).to(self.hparams.device) for i in locality_image]
             
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
            
        return requests


# if __name__ == "__main__":
#
#     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
#
#     editor.edit(
#         prompts='What university did Watts Humphrey attend?',
#         ground_truth='Illinois Institute of Technology',
#         target_new='University of Michigan'
#     )
#
#     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')


