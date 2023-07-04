from collections import defaultdict
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import nethook


class LogitLens:
    """
    Applies the LM head at the output of each hidden layer, then analyzes the
    resultant token probability distribution.

    Only works when hooking outputs of *one* individual generation.

    Inspiration: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

    Warning: when running multiple times (e.g. generation), will return
    outputs _only_ for the last processing step.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        layer_module_tmp: str,
        ln_f_module: str,
        lm_head_module: str,
        disabled: bool = False,
    ):
        self.disabled = disabled
        self.model, self.tok = model, tok
        self.n_layers = self.model.config.n_layer

        self.lm_head, self.ln_f = (
            nethook.get_module(model, lm_head_module),
            nethook.get_module(model, ln_f_module),
        )

        self.output: Optional[Dict] = None
        self.td: Optional[nethook.TraceDict] = None
        self.trace_layers = [
            layer_module_tmp.format(layer) for layer in range(self.n_layers)
        ]

    def __enter__(self):
        if not self.disabled:
            self.td = nethook.TraceDict(
                self.model,
                self.trace_layers,
                retain_input=False,
                retain_output=True,
            )
            self.td.__enter__()

    def __exit__(self, *args):
        if self.disabled:
            return
        self.td.__exit__(*args)

        self.output = {layer: [] for layer in range(self.n_layers)}

        with torch.no_grad():
            for layer, (_, t) in enumerate(self.td.items()):
                cur_out = t.output[0]
                assert (
                    cur_out.size(0) == 1
                ), "Make sure you're only running LogitLens on single generations only."

                self.output[layer] = torch.softmax(
                    self.lm_head(self.ln_f(cur_out[:, -1, :])), dim=1
                )

        return self.output

    def pprint(self, k=5):
        to_print = defaultdict(list)

        for layer, pred in self.output.items():
            rets = torch.topk(pred[0], k)
            for i in range(k):
                to_print[layer].append(
                    (
                        self.tok.decode(rets[1][i]),
                        round(rets[0][i].item() * 1e2) / 1e2,
                    )
                )

        print(
            "\n".join(
                [
                    f"{layer}: {[(el[0], round(el[1] * 1e2)) for el in to_print[layer]]}"
                    for layer in range(self.n_layers)
                ]
            )
        )
