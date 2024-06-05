import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util.globals import *

from .layer_stats import layer_stats
from .r_rome_hparams import R_ROMEHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to(f"cuda:{hparams.device}")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: R_ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")

        if hparams.original_implementation or hparams.enable_random_prefix_keys:
            cur_repr = repr_tools.get_reprs_at_word_tokens(
                context_templates=[
                    templ.format(request["prompt"]) for templ in context_templates
                ],
                words=[word for _ in range(len(context_templates))],
                subtoken=hparams.fact_token[len("subject_") :],
                **word_repr_args,
            ).mean(0)
        else:
            cur_repr = repr_tools.get_reprs_at_word_tokens(
                subtoken=hparams.fact_token[len("subject_") :],
                context_templates=[request["prompt"]],
                words=[word],
                **word_repr_args,
            ).squeeze()

    elif hparams.fact_token == "last":
        if hparams.enable_prompt_keys:
            raise ValueError(
                "Last token projection not supported with prompt_keys. Use subject_ instead."
            )
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            hparams=hparams,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()