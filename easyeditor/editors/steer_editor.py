from typing import Optional, Union, List, Tuple, Dict
from time import time
from PIL import Image
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from ..models.dola import generate as dola_generate
from ..models.deco import generate as deco_generate

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from ..util.globals import *
from ..evaluate import compute_safety_edit_quality, ccks_compute_safety_edit_quality
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


class SteerEditor:
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.alg_name = hparams.alg_name
        self.modal_type = None # llm or mllm 
        
        make_logs()

        LOG.info("Instantiating model")
        

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            if 'llama' in self.model_name.lower():
                self.model = LlamaForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
                self.modal_type = 'llm'
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
                self.modal_type = 'llm'
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id    
                self.modal_type = 'llm'
            elif 'llava' in self.model_name.lower():
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoProcessor.from_pretrained(self.model_name)
                self.modal_type = 'mllm'
            elif 'blip' in self.model_name.lower():
                self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = InstructBlipProcessor.from_pretrained(self.model_name)
                self.modal_type = 'mllm'
            else:
                raise NotImplementedError
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        self.device = torch.device(f'cuda:{hparams.device}')
        
        if self.modal_type == 'llm':
            self.lm_head = self.model.get_output_embeddings()
            self.norm = self.model.model.norm
        else:
            self.lm_head = self.model.language_model.get_output_embeddings()
            self.norm = self.model.language_model.model.norm
    
    
    def generate(self, 
                 input_text: str = None, 
                 input_image: str = None,
                 temperature: float = None,
                 **model_kwargs
                ):
        assert (input_image is not None and self.modal_type == 'mllm') or (input_image is None and self.modal_type == 'llm'), print('Error: llm cannot process image input or input_image is None.')
        
        if self.modal_type == 'llm':
            inputs = self.tok(input_text, return_tensors="pt").to(self.device)
        else:
            input_image = Image.open(input_image).convert("RGB")
            inputs = self.tok(images=input_image, text=input_text, return_tensors="pt").to(self.device)
            
        # constrained decoding methods: dola, deco, opera, vcd
        if self.alg_name == 'dola':
            transformers.generation.utils.GenerationMixin.generate = dola_generate
            outputs = self.model.generate(
                                        **inputs, 
                                        do_sample=False, 
                                        dola_layers=self.hparams.dola_layers,
                                        final_layer=self.hparams.final_layer,
                                        lm_head=self.lm_head,
                                        **model_kwargs
                                        )
        elif self.alg_name == 'deco':
            transformers.generation.utils.GenerationMixin.generate = deco_generate
            outputs = self.model.generate(
                                        **inputs, 
                                        do_sample=True if temperature > 0 else False,
                                        alpha=self.hparams.alpha,
                                        threshold_top_p=self.hparams.threshold_top_p,
                                        threshold_top_k=self.hparams.threshold_top_k,
                                        early_exit_layers=self.hparams.early_exit_layers,
                                        lm_head=self.lm_head,
                                        norm=self.norm,
                                        **model_kwargs
                                        )
        # base decoding methods
        else:
            outputs = self.model.generate(
                                        **inputs, 
                                        do_sample=True if temperature > 0 else False,
                                        **model_kwargs
                                        )
        
        
        output_text = self.tok.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return output_text
        
        
        