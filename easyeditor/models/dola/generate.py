import numpy as np
from torch.nn import functional as F
from transformers.generation.utils import (
    LogitsProcessorList, 
    StoppingCriteriaList, 
    GenerationConfig, 
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateOutput
)

import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.cache_utils import StaticCache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}



logger = logging.get_logger(__name__)

@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    dola_layers: Union[str, List[int]] = None,
    final_layer: int = None,
    lm_head: nn.Module = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if generation_config.cache_implementation == "static":
            if model_kwargs.get("past_key_values", False) is not False:
                raise ValueError(
                    "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                )
            cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
            if not callable(getattr(self, "_setup_cache", None)):
                raise ValueError(
                    "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                    " Make sure it has a `_setup_cache` function."
                )
            self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    def _dola_decoding(
        input_ids: torch.LongTensor,
        dola_layers: Union[str, List[int]],
        final_layer: int,
        lm_head: nn.Module,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: "BaseStreamer",
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
        used for decoder-only text models.
        The method is based on the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
        Models" (https://arxiv.org/abs/2309.03883) in ICLR 2024.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            dola_layers (`Union[str, List[int]]`):
                The candidate layers used in contrasting layers of DoLa. It can be either 1) 'low' or 'high', which
                means the lower part or higher part of the model layers, respectively, or 2) a list of layer indices
                to be used for candidate layers. The 0-th layer is the word embedding layer of the model.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        if self.config.is_encoder_decoder:
            raise ValueError("DoLa decoding is only available for decoder-only models.")
        # init values

        # pad_token_id = generation_config._pad_token_tensor
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        
        this_peer_finished = False

        # prepare layers for DoLa decoding
        # final_layer = self.config.get_text_config().num_hidden_layers
        final_layer = final_layer
        # if the model has tied word embeddings, we skip the word embeddings (0-th) layer and start from the 2nd layer,
        # as the early exit from word embeddings will become identity function
        # if the model is really shallow (<=2 layers), we use the 1st layer if it's not the final layer and the 0-th
        # layer otherwise. Notice that DoLa does not help shallow models much.
        if not self.config.tie_word_embeddings:
            start_layer = 0
        elif final_layer > 2:
            start_layer = 2
        elif final_layer == 2:
            start_layer = 1
        else:
            start_layer = 0

        # For `N`-layer models with `N <= 40` layers, the layers of `range(0, N // 2, 2)` and `range(N // 2, N, 2)`
        # are used for `'low'` and `'high'` layers, respectively.
        # For models with `N > 40` layers, the layers of `range(0, 20, 2)` and `range(N - 20, N, 2)` are used for
        # `'low'` and `'high'` layers, respectively.
        if isinstance(dola_layers, str) and dola_layers == "low":
            if start_layer == final_layer // 2:
                candidate_premature_layers = [start_layer]
            else:
                candidate_premature_layers = (
                    list(range(start_layer, final_layer // 2, 2))
                    if final_layer <= 40
                    else list(range(start_layer, 20, 2))
                )
        elif isinstance(dola_layers, str) and dola_layers == "high":
            candidate_premature_layers = (
                list(range(final_layer // 2, final_layer, 2))
                if final_layer <= 40
                else list(range(final_layer - 20, final_layer, 2))
            )
        # Set the `dola_layers` to a list of integers for layer indices to contrast manually specified layers.
        elif isinstance(dola_layers, list):
            candidate_premature_layers = [i for i in dola_layers if i < final_layer]
        else:
            raise ValueError("dola_layers must be either 'low', 'high' or a list of integers.")

        lm_head = lm_head
        if lm_head is None:
            raise ValueError("DoLa is not supported for models that don't have output embeddings.")

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            # .float() is needed to retain precision for later logits manipulations
            final_layer_next_token_logits = outputs.logits[:, -1, :].detach().clone().float()
            final_logits = outputs.logits[:, -1, :].float()
            candidate_premature_logits = {}
            for candidate_premature_layer in candidate_premature_layers:
                candidate_premature_logits[candidate_premature_layer] = lm_head(
                    outputs.hidden_states[candidate_premature_layer][:, -1, :]
                ).to(final_logits.device)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = _dola_select_contrast(
                candidate_premature_layers, candidate_premature_logits, final_logits
            )
            next_token_logits = next_token_logits.to(input_ids.device)
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (final_layer_next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:  # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:  # argmax
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        
        
    def _relative_top_filter(
        scores: torch.FloatTensor,
        baseline_scores: torch.FloatTensor,
        relative_top: float = 0.1,
        filter_value: float = -float("Inf"),
        base_filter_value=-1e-3,
        min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
        """
        Reference: https://github.com/XiangLi1999/ContrastiveDecoding/blob/170e9142e92159c1237d731e240f5eb14aabf428/transformers/src/transformers/generation_logits_process.py#L235
        Apply filtering to only keep tokens with a probability above a certain threshold. The threshold is defined as `relative_top` * max probability in the distribution.
        """
        scores_normalized = scores.log_softmax(dim=-1)
        baseline_scores_normalized = baseline_scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        baseline_scores_normalized[scores_normalized < probs_thresh] = base_filter_value
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized, baseline_scores_normalized


    def _dola_select_contrast(
        candidate_premature_layers: List[int],
        candidate_premature_logits: Dict[int, torch.FloatTensor],
        final_logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if len(candidate_premature_layers) == 1:
            base_logits = candidate_premature_logits[candidate_premature_layers[0]]
            final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
            logits = final_logits - base_logits
            return logits

        # 1. Stacking all premature_layers into a new dimension
        stacked_premature_layers = torch.stack([candidate_premature_logits[i] for i in candidate_premature_layers], dim=0)

        # 2. Calculate the softmax values for mature_layer and all premature_layers
        # shape: (batch_size, vocab_size)
        softmax_mature_layer = F.softmax(final_logits, dim=-1)
        # shape: (num_premature_layers, batch_size, vocab_size)
        softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)

        # 3. Calculate the average distribution
        # shape: (num_premature_layers, batch_size, vocab_size)
        avg_dist = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)

        # 4. Calculate log-softmax for the KL divergence
        # shape: (batch_size, vocab_size)
        log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1)
        # shape: (num_premature_layers, batch_size, vocab_size)
        log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)

        # 5. Calculate the KL divergences and then the JS divergences
        # shape: (num_premature_layers, batch_size)
        kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], avg_dist, reduction="none").mean(-1)
        # shape: (num_premature_layers, batch_size)
        kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1)
        js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

        # 6. Reduce the batchmean
        js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
        premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]

        base_logits = candidate_premature_logits[premature_layer]
        final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
        logits = final_logits - base_logits
        return logits
    
    
    
    
    result = _dola_decoding(
        input_ids=input_ids,
        dola_layers=dola_layers,
        final_layer=final_layer,
        lm_head=lm_head,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )

    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if not callable(getattr(self, "_reset_cache", None)):
            raise ValueError(
                "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                " Make sure this model implements a `_reset_cache` function."
            )
        self._reset_cache()

    return result