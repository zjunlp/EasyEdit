import os
from copy import deepcopy
from typing import Dict, List

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *

from ...trainer import EFK
from .efk_hparams import KEHyperParams


class EfkRewriteExecutor:

    def __init__(self) -> None:
        self.is_init = False

    def init_model(self, model, tok, params):
        # train_ds = (
        #     "counterfact-" if params.counterfact else ("zsre-" if params.zsre else "")
        # )

        # modelcode = "gpt2xl" if params.model_name == "gpt2-xl" else "gpt-j-6b"
        # model_filename = f"efk-{params.n_toks}tok-{train_ds}gpt2-xl.pt"
        # model_dir = "baselines/efk/weights"
        #
        # os.makedirs(model_dir, exist_ok=True)
        # if not os.path.isfile(f"{model_dir}/{model_filename}"):
        #     torch.hub.download_url_to_file(
        #         f"{REMOTE_ROOT_URL}/data/weights/{model_filename}",
        #         f"{model_dir}/{model_filename}",
        #     )
        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Load the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        # add_padding(self.tokenizer, self.model)

        # Load the trained EFK model
        self.alg = EFK(self.model, params, lambda: deepcopy(self.model))
        d = torch.load(params.archive)
        self.alg.load_state_dict(d["model"])
        self.alg.to(torch.device(f'cuda:{params.device}'))
        self.is_init = True

    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: KEHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs
    ):
        """
        Processes a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns an edited GPT model.
        """

        if copy:
            model = deepcopy(model)

        if not self.is_init:
            self.init_model(model, tok, hparams)

        for request in requests:
            self.init_model(model, tok, hparams)
            request_rewrite = deepcopy(request)

            target = " " + request_rewrite["target_new"]
            sentence = (
                request_rewrite["prompt"] + target
            )
            target_tokens = self.tokenizer(target)["input_ids"]
            tokens = torch.tensor(self.tokenizer(sentence)["input_ids"])[None]
            # label_tokens = tokens.clone()
            # label_tokens[0][: -len(target_tokens)] = -100
            edit_inner = dict(
                input_ids=tokens.to(f"cuda:{hparams.device}"),
                attention_mask=torch.ones_like(tokens).to(f"cuda:{hparams.device}"),
                labels=torch.tensor(target_tokens).unsqueeze(0).to(f"cuda:{hparams.device}"),
            )
            cond = dict(
                input_ids=tokens.to(f"cuda:{hparams.device}"),
                attention_mask=torch.ones_like(tokens).to(f"cuda:{hparams.device}"),
            )

            weights_copy = {}
            if return_orig_weights:
                for k, v in model.named_parameters():
                    if k not in weights_copy:
                        weights_copy[k] = (
                            v.detach().to(f"cuda:{hparams.device}")
                        )

            edited_model, _ = self.alg.edit(edit_inner, cond, detach_history=False)
            model = edited_model.model

        if not keep_original_weight:
            weights_copy = {}

        return model, weights_copy
