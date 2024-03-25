from .editor import BaseEditor
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import pdb
import random

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from ..evaluate import (
    compute_per_ike_metric,
    compute_per_metric
    )
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


class PerEditor:
    """Personality Editor for IKE & MEND"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = PER_ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = 0 if self.tok.pad_token_id is None else self.tok.pad_token_id
                self.tok.bos_token_id = 1
            if "gpt" in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
                self.tok.add_special_tokens({'sep_token': '</s>'})
                self.model.resize_token_embeddings(len(self.tok))
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
            self.device = hparams.device

        self.hparams = hparams
        
        
    def edit_dataset(self, ds: Dataset, keep_original_weight=False, verbose=True):
        """edit for IKE in Personality Dataset"""
        # Make Sure dataset supportedxiao
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in PER_DS_DICT.values()]) > 0, print(f'DataSet {ds} not supported yet.')
                
        all_metrics = []
        collate_fn = ds.collate_gpt_fn
        for i, request in tqdm(enumerate(ds), desc='Editing dataset', total=len(ds)):
            start = time()
            
            if self.alg_name == 'IKE':
                edited_model, weights_copy = self.model, {}
                outer_idx = (i + 1) % len(ds)
                loc_case = ds[outer_idx]
                example = self.apply_algo(request=request, loc_request=loc_case, tokenizer=self.tok, device=self.device)
                
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                }
                metrics.update(compute_per_ike_metric(example=example, model=edited_model,tok=self.tok, device=self.device, test_generation=True))
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['ent']} -> {request['target_personality']}  \n {metrics}"
                    )

                all_metrics.append(metrics)
                
            else:
                example = collate_fn([request])
                edited_model, weights_copy = self.apply_algo(
                    request=example,
                    model=self.model,
                    tok=self.tok,
                    hparams=self.hparams,
                    device=self.device,
                )
                
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                metrics = {
                    'case_id': i,
                    "time": exec_time,
                }
                
                metrics.update(compute_per_metric(example=example, model=self.model, edited_model=edited_model, tok=self.tok, device=self.device, test_generation=True))
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['ent']} -> {request['target_personality']}  \n {metrics}"
                    )
                    
                all_metrics.append(metrics)


        return all_metrics, edited_model, weights_copy
    
    