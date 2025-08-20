import os
from copy import deepcopy
from typing import Dict, List, Any, Tuple

import hydra
import torch
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *

from .ULTRAEDIT import ULTRAEDIT
from .ultraedit_hparams import UltraEditHyperParams

class UltraEditRewriteExecutor:
    def __init__(self):
        self.is_init = False

    def init_model(self, model, tok, params: UltraEditHyperParams):

        self.model = model
        self.tokenizer = tok

        # Load the trained MEND model
        self.alg = ULTRAEDIT(self.model, params, lambda: deepcopy(self.model))
        if params.model_parallel:
            self.alg.lifelong_normalizer.to(deque(self.alg.model.parameters(), maxlen=1)[0].device)
        else:
            self.alg.to(torch.device(f'cuda:{params.device}'))


    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: UltraEditHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """

        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model
        assert len(requests) >= hparams.batch_size, "The number of requests must be greater than or equal to the value of batch_size."
        # Define i/o
        requests = requests[:hparams.batch_size]
        batchs = []
        for i in range(hparams.batch_size // hparams.batch_size_once):
            batch = requests[i * hparams.batch_size_once : (i+1)*hparams.batch_size_once]
            targets = [
                (" " if request["target_new"][0] != " " else "")
                + request["target_new"]
                for request in batch
            ]
            sentences = [
                request["prompt"] + targets[i]
                for i, request in enumerate(batch)
            ]

            # Tokenize
            sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
                f"cuda:{hparams.device}"
            )
            target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(
                f"cuda:{hparams.device}"
            )

            # Define labels
            label_tok = deepcopy(sent_tok["input_ids"])
            for i in range(label_tok.size(0)):
                target_len = target_tok["attention_mask"][i].sum()
                padding_len = (
                    sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
                )
                label_tok[i][: -target_len - padding_len] = -100
                label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

            edit_inner = dict(
                input_ids=sent_tok["input_ids"],
                attention_mask=sent_tok["attention_mask"],
                labels=target_tok['input_ids'],
            )

            batchs.append(edit_inner)
        # Run M
        batchs = sorted(
            batchs,
            key=lambda x: torch.sum(x['attention_mask']).item(),
            reverse=True
        )        
        module_kv_map = self.alg.cache(batchs)
        param_shifts = self.alg.predict_param_shifts(module_kv_map)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in hparams.inner_params:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()
        self.alg.edit_model(param_shifts, False)

        return self.alg.model, weights_copy
 