from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from .utils import (
    get_parameter,
    set_parameter,
    TracerDict,
)
import torch

class SimIE:
    def __init__(self, lamHyper: float, init: bool, solver: str="LU"):
        self.lamHyper = lamHyper
        self.init = init
        self.solver = solver

    def initialization(self, model_name, init_weights, device, fast=True):
        self.device = f"cuda:{device}"
        self.init_weights_copy = {n: p.clone().cpu() for n, p in init_weights.items()}
        self.weights_copy = {n: p.clone().cpu() for n, p in init_weights.items()}
        if "llama" in model_name.lower() or "gpt-j-6b" in model_name.lower() or "mistral" in model_name.lower():
            self.matrix_P = {n: self.lamHyper * torch.eye(p.shape[1]).to(dtype=p.dtype) for n, p in init_weights.items()}
            self.transpose = False
        elif "gpt2-xl" in model_name.lower():
            self.matrix_P = {n: self.lamHyper * torch.eye(p.shape[0]).to(dtype=p.dtype) for n, p in init_weights.items()}
            self.transpose = True
        else:
            raise ValueError("Need to check the matrix P initialization")
        self.config = {
            "token": "mask",
            "inner_params": []
        }
        for n, _ in init_weights.items():
            self.config["inner_params"].append(n.replace(".weight", ""))

        self.fast = False
        if fast:
            self.fast = True
            self.init_weights_copy = {n: p.to(self.device) for n, p in self.init_weights_copy.items()}
            self.weights_copy = {n: p.to(self.device) for n, p in self.weights_copy.items()}
            self.matrix_P = {n: p.to(self.device) for n, p in self.matrix_P.items()}

    @staticmethod
    def solve_eqn(A, B, solver="LU", rcond=None):

        def solve_with_lu(A, B):
            return torch.linalg.solve(A.T, B.T).T

        if solver == "LU":
            result = solve_with_lu(A, B)
        elif solver == "Cholesky":
            try:
                L = torch.linalg.cholesky(A.T)
                result = torch.cholesky_solve(B.T, L).T
            except RuntimeError as e:
                print(f"Cholesky solve failed: {e}. Falling back to LU decomposition.")
                result = solve_with_lu(A, B)
        elif solver == "SVD":
            try:
                if rcond is not None:
                    result = B @ torch.linalg.pinv(A, rcond=rcond, hermitian=True)
                else:
                    result = B @ torch.linalg.pinv(A, hermitian=True)
            except RuntimeError as e:
                print(f"SVD solve failed: {e}. Falling back to LU decomposition.")
                result = solve_with_lu(A, B)
        else:
            raise ValueError("Invalid solver")
        return result

    def reset_parameter(self, model):
        return set_parameter(model, self.init_weights_copy, self.device)

    def cache(self, model, requests, tok):

        # define batch
        batch_size = 1
        n_edits = len(requests)

        batchs = []
        for i in range(n_edits // batch_size):
            batch = requests[i * batch_size : (i+1) * batch_size]
            targets = [
                (" " if request["target_new"][0] != " " else "")
                + request["target_new"]
                for request in batch
            ]
            sentences = [
                request["prompt"] + targets[i]
                for i, request in enumerate(batch)
            ]

            # tokenize
            sent_tok = tok(sentences, padding=True, return_tensors="pt").to(self.device)
            target_tok = tok(targets, padding=True, return_tensors="pt").to(self.device)

            # define labels
            label_tok = deepcopy(sent_tok["input_ids"])
            for i in range(label_tok.size(0)):
                target_len = target_tok["attention_mask"][i].sum()
                padding_len = (
                    sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
                )
                label_tok[i][: -target_len - padding_len] = -100
                label_tok[i][label_tok[i] == tok.pad_token_id] = -100

            edit_inner = dict(
                input_ids=sent_tok["input_ids"],
                attention_mask=sent_tok["attention_mask"],
                labels=target_tok['input_ids'],
            )
            
            batchs.append(edit_inner)

        # cache keys
        with torch.no_grad():
            keys_cache = {}
            for idx, t in enumerate(batchs):
                with TracerDict(
                    model,
                    self.config,
                    t
                ) as tr:
                    _ = model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])

                for module_idx, module_name in enumerate(self.config["inner_params"]):
                    keys = tr[module_name].keys.to(torch.float32).to(self.device).clone() 
                    keys_cache.setdefault(module_name+".weight", {}).update({idx: {'keys': keys}})
        return keys_cache
    
    def update(self, edited_model, keys_cache):

        for module_name, batch_data in keys_cache.items():
            keys_list = []
            
            for idx, data in batch_data.items():
                keys_list.append(data['keys'])
            
            # combine keys
            key_all = torch.cat(keys_list, dim=0)
            if self.fast:
                self.matrix_P[module_name] += (key_all.T @ key_all)
            else:
                self.matrix_P[module_name] += (key_all.T @ key_all).cpu()
            with torch.no_grad():
                weights_copy_gpu = self.weights_copy[module_name].to(self.device)
                param = get_parameter(edited_model, module_name)
                if self.transpose:
                    mat = (param - weights_copy_gpu).T @ key_all.T @ key_all
                    delta_par = SimIE.solve_eqn(self.matrix_P[module_name].to(self.device), mat, self.solver).T
                elif not self.transpose:
                    mat = (param - weights_copy_gpu) @ key_all.T @ key_all
                    delta_par = SimIE.solve_eqn(self.matrix_P[module_name].to(self.device), mat, self.solver)
                weights_copy_gpu += delta_par
                param[...] = weights_copy_gpu
                if self.fast:
                    self.weights_copy[module_name] = weights_copy_gpu.clone()
                else:
                    self.weights_copy[module_name] = weights_copy_gpu.cpu()
            # del key_all, weights_copy_gpu, mat, delta_par

        return edited_model
    
    def ideal_editor(self, edited_model, keys_cache, run=None):
        if not hasattr(self, 'KKT'):
            if self.transpose:
                self.KKT = {n: torch.zeros(p.shape[0],p.shape[0]).to(dtype=p.dtype) for n, p in self.init_weights_copy.items()}
                self.BKT = {n: torch.zeros(p.shape[1],p.shape[0]).to(dtype=p.dtype) for n, p in self.init_weights_copy.items()}
            else:
                self.KKT = {n: torch.zeros(p.shape[1],p.shape[1]).to(dtype=p.dtype) for n, p in self.init_weights_copy.items()}
                self.BKT = {n: torch.zeros(p.shape[0],p.shape[1]).to(dtype=p.dtype) for n, p in self.init_weights_copy.items()}

        for module_name, batch_data in keys_cache.items():
            keys_list = []   
            for idx, data in batch_data.items():
                keys_list.append(data['keys'])
            # combine keys
            key_all = torch.cat(keys_list, dim=0)
            self.KKT[module_name] += (key_all.T @ key_all).cpu()
            parameter_diff = get_parameter(edited_model, module_name) - self.init_weights_copy[module_name]
            if self.transpose:
                self.BKT[module_name] += (parameter_diff.T @ key_all.T @ key_all).cpu()
            else:
                self.BKT[module_name] += (parameter_diff @ key_all.T @ key_all).cpu()
        
        if run is not None:
            ideal = {
                "KKT": self.KKT,
                "BKT": self.BKT
            }
            torch.save(ideal, f"outputs/{run.config.data_type}_{run.config.model_name}_{run.config.editing_method}_ideal_editor.pth")


    