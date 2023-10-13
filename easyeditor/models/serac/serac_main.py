import os
from copy import deepcopy
from typing import Dict, List

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *

from ...trainer import SERAC, SERAC_MULTI
from .serac_hparams import SERACHparams


class SeracRewriteExecutor:
    def __init__(self):
        self.is_init = False

    def init_model(self, model, tok, params: SERACHparams):

        assert params.archive is not None or print(f'Training weights Needed....')

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        def set_padding():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
        set_padding()

        # Load the trained MEND model
        self.alg = SERAC(self.model, deepcopy(params), lambda: deepcopy(self.model))
        d = torch.load(params.archive, map_location='cpu')
        self.alg.load_state_dict(d["model"], False)
        # self.alg.to(torch.device(f'cuda:{params.device}'))
        self.alg.replacement.to(torch.device(f'cuda:{params.device}'))
        self.alg.classifier.to(torch.device(f'cuda:{params.device}'))

        self.is_init = True

    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: SERACHparams,
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

        # Define i/o
        targets = [
            (" " if request["target_new"][0] != " " else "")
            + request["target_new"]
            for request in requests
        ]
        sentences = [
            request["prompt"] + targets[i]
            for i, request in enumerate(requests)
        ]
        #
        # # Tokenize
        sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
            f"cuda:{hparams.device}"
        )
        label_tok = self.tokenizer([request["target_new"] for request in requests],
                                    padding=True,
                                    return_tensors="pt"
                                    ).to(f"cuda:{hparams.device}")
        #
        # # label_tok = deepcopy(sent_tok["input_ids"])
        # # for i in range(label_tok.size(0)):
        # #     target_len = target_tok["attention_mask"][i].sum()
        # #     padding_len = (
        # #         sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
        # #     )
        # #     label_tok[i][: -target_len - padding_len] = -100
        # #     label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100
        #
        # # Run MEND
        # edit_inner = dict(
        #     input_ids=sent_tok["input_ids"],
        #     attention_mask=sent_tok["attention_mask"],
        #     labels=label_tok["input_ids"],
        # )
        # cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        # new_model = None
        #
        #
        # new_model, model_info = self.alg.edit(edit_inner, cond)

        # targets = [
        #     (" " if request["target_new"][0] != " " else "")
        #     + request["target_new"]
        #     for request in requests
        # ]
        # target_tok = self.tokenizer(targets,
        #                             truncation=True,
        #                             max_length=hparams.max_length)["input_ids"]
        #
        # sentences = [
        #     [request['prompt'] + self.tokenizer.decode(target_tok[i][:j]) for j in range(len(target_tok[i]))]
        #     for i, request in enumerate(requests)
        # ]
        #
        # sentences = [sentence for sentences_ in sentences for sentence in sentences_]
        #
        # targets = [
        #     [self.tokenizer.decode(target_tok[i][j]) for j in range(len(target_tok[i]))]
        #     for i, request in enumerate(requests)
        # ]
        #
        # targets = [target for targets_ in targets for target in targets_]

        # label_tok = self.tokenizer(targets,
        #                             padding=True,
        #                             return_tensors="pt"
        #                             ).to(f"cuda:{hparams.device}")

        # Tokenize
        # sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
        #     f"cuda:{hparams.device}"
        # )

        # label_tok = deepcopy(sent_tok["input_ids"])
        # for i in range(label_tok.size(0)):
        #     target_len = target_tok["attention_mask"][i].sum()
        #     padding_len = (
        #         sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
        #     )
        #     label_tok[i][: -target_len - padding_len] = -100
        #     label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=label_tok["input_ids"],
        )
        cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        new_model = None

        new_model, model_info = self.alg.edit(edit_inner, cond)
        # factors = {
        #     k + "." + n: v.detach().cpu().numpy()
        #     for k, pair in model_info["factors"].items()
        #     for n, v in zip("uv", pair)
        # }
        # # Also keep these learned LRs.
        # factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()
        #
        # # Edit!
        # d = factors
        # torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        # eli = 0
        # edit_lrs = torch_factors["edit_lrs"]
        #
        # with torch.no_grad():
        #     for n, p in model.named_parameters():
        #         uname, vname = f"{n}.u", f"{n}.v"
        #         if uname in torch_factors:
        #             if return_orig_weights and n not in weights_copy:
        #                 weights_copy[n] = p.detach().clone()
        #
        #             if "gpt2" in hparams.model_name:
        #                 delta = torch_factors[uname].t() @ torch_factors[vname]
        #             elif "gpt-j-6B" in hparams.model_name:
        #                 delta = torch_factors[vname].t() @ torch_factors[uname]
        #             else:
        #                 raise ValueError("Unknown model")
        #             p.add_((delta * edit_lrs[eli] * hparams.lr_scale).to(p.device))
        #             eli += 1

        if keep_original_weight:
            self.alg.cache_labels = self.alg.cache_labels[-1:]
            self.alg.cache_inputs = self.alg.cache_inputs[-1:]

        return new_model, {}
    
class SeracMultimodalRewriteExecutor(SeracRewriteExecutor):
    def __init__(self):
        super().__init__()

    def init_model(self, model, tok, params: SERACHparams):

        assert params.archive is not None or print(f'Training weights Needed....')

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        def set_padding():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
        set_padding()

        # Load the trained MEND model
        self.alg = SERAC_MULTI(self.model, params, lambda: deepcopy(self.model))
        d = torch.load(params.archive, map_location='cpu')
        self.alg.load_state_dict(d["model"], False)
        self.alg.to(torch.device(f'cuda:{params.device}'))
        self.alg.replacement.to(torch.device(f'cuda:{params.device}'))
        self.alg.classifier.to(torch.device(f'cuda:{params.device}'))

        self.is_init = True

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: SERACHparams,
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

        # Define i/o
        src = [request["prompt"] for request in requests]
        trg = [
            (" " if request["target"][0] != " " else "")
            + request["target"]
            for request in requests
        ]
        image = [request["image"] for request in requests]
        image = torch.stack(image, dim=0)
        text_input = [s + t for s, t in zip(src, trg)]
        labels = trg
        if hparams.model_name == "minigpt4":
            prompts_len = [len(tok.encode(s, add_special_tokens=False)) for s in src]
        else:
            prompts_len = [len(tok.encode(s)) for s in src]

        # Run MEND
        edit_inner = dict(
            image=image,
            text_input=text_input,
            labels=labels,
            prompts_len=prompts_len
        )
        new_model = None


        new_model, model_info = self.alg.edit(edit_inner,)
        # factors = {
        #     k + "." + n: v.detach().cpu().numpy()
        #     for k, pair in model_info["factors"].items()
        #     for n, v in zip("uv", pair)
        # }
        # # Also keep these learned LRs.
        # factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()
        #
        # # Edit!
        # d = factors
        # torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        # eli = 0
        # edit_lrs = torch_factors["edit_lrs"]
        #
        # with torch.no_grad():
        #     for n, p in model.named_parameters():
        #         uname, vname = f"{n}.u", f"{n}.v"
        #         if uname in torch_factors:
        #             if return_orig_weights and n not in weights_copy:
        #                 weights_copy[n] = p.detach().clone()
        #
        #             if "gpt2" in hparams.model_name:
        #                 delta = torch_factors[uname].t() @ torch_factors[vname]
        #             elif "gpt-j-6B" in hparams.model_name:
        #                 delta = torch_factors[vname].t() @ torch_factors[uname]
        #             else:
        #                 raise ValueError("Unknown model")
        #             p.add_((delta * edit_lrs[eli] * hparams.lr_scale).to(p.device))
        #             eli += 1

        if keep_original_weight:
            self.alg.cache_labels = self.alg.cache_labels[-1:]
            self.alg.cache_inputs = self.alg.cache_inputs[-1:]

        return new_model, {}
