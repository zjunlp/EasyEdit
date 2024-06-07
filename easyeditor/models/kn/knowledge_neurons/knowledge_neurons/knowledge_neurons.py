 # main knowledge neurons class
import collections
import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .patch import *


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif 'gptj' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.fc_in"
            self.output_ff_attr = "mlp.fc_out.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif "gpt2" == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte"
        elif 'llama' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'baichuan' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif "t5" == model_type:
            self.transformer_layers_attr = "decoder.block"
            self.input_ff_attr = "layer.2.DenseReluDense.wi"
            self.output_ff_attr = "layer.2.DenseReluDense.wo.weight"
            self.word_embeddings_attr = "shared.weight"
        elif 'chatglm2' == model_type:
            self.transformer_layers_attr = "transformer.encoder.layers"
            self.input_ff_attr = "mlp.dense_4h_to_h"
            self.output_ff_attr = "mlp.dense_h_to_4h.weight"
            self.word_embeddings_attr = "transformer.embedding.word_embeddings"
        elif 'internlm' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'qwen2' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif 'qwen' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.w1"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif 'mistral' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.model_type == 't5':
            target_input = self.tokenizer(target, return_tensors='pt').to(self.device)
            encoded_input['decoder_input_ids'] = target_input['input_ids']
            encoded_input['decoder_attention_mask'] = target_input['attention_mask']
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        elif self.model_type == 't5':
            mask_idx = list(range(encoded_input['decoder_input_ids'].size(1)))
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            if "qwen" in self.model_type or "gpt" in self.model_type or 't5' in self.model_type or 'llama' in self.model_type:
                target = self.tokenizer.encode(target)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("qwen" in self.model_type or "gpt" in self.model_type or 'llama' in self.model_type) else 1
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i] if n_sampling_steps > 1 else target_label
            # print(probs.shape)
            # gt_prob = probs[:, target_idx].item()
            # print(target_idx)
            if self.model_type == 't5':
                for q, target_idx_ in enumerate(target_idx):
                    gt_prob_= probs[:, q, target_idx_]
                    all_gt_probs.append(gt_prob_)

                    argmax_prob, argmax_id = [i.item() for i in probs[:,q,:].max(dim=-1)]
                    argmax_tokens.append(argmax_id)
                    argmax_str = self.tokenizer.decode([argmax_id])
                    all_argmax_probs.append(argmax_prob)

                    argmax_completion_str += argmax_str
            else:
                gt_prob = probs[:, target_idx]
                # print(gt_prob.shape)
                all_gt_probs.append(gt_prob)

                # get info about argmax completion
                argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
                argmax_tokens.append(argmax_id)
                argmax_str = self.tokenizer.decode([argmax_id])
                all_argmax_probs.append(argmax_prob)

                prompt += argmax_str
                argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")
    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
            )
            scores.append(layer_scores)
        scores = [score.to(self.device) for score in scores]
        return torch.stack(scores)

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), "Provide one and only one of threshold / adaptive_threshold / percentile"

        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            coarse_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
            if self.model_type == 't5' and len(coarse_neurons) > 0 and len(coarse_neurons[0]) == 3:
                coarse_neurons = list(set([(layer_idx, neuron_idx) for layer_idx, _, neuron_idx in coarse_neurons]))
            return coarse_neurons
        s = attribution_scores.flatten().detach().cpu().numpy()
        return (
            torch.nonzero(attribution_scores > np.percentile(s, percentile))
            .cpu()
            .tolist()
        )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
        refine: bool = False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = [
            self.get_coarse_neurons(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                adaptive_threshold=coarse_adaptive_threshold,
                threshold=coarse_threshold,
                percentile=coarse_percentile,
                pbar=False,
            )
            for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
            )
        ]
        if negative_examples is not None:
            negative_neurons = [
                self.get_coarse_neurons(
                    negative_example,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
                for negative_example in tqdm(
                    negative_examples,
                    desc="Getting coarse neurons for negative examples",
                    disable=quiet,
                )
            ]
        if not quiet:
            total_coarse_neurons = sum(len(i) for i in coarse_neurons)
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        if refine:
            refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
        else:
            refined_neurons = [list(neuron) for neuron, count in c.items()]
        # filter out neurons that are in the negative examples
        if negative_examples is not None and False:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        if not quiet:
            total_refined_neurons = len(refined_neurons)
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("qwen" in self.model_type or "gpt" in self.model_type or 'llama' in self.model_type) else 1
        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0 and ("qwen" in self.model_type or self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                    if self.model_type == 't5':
                        inputs["decoder_input_ids"] = einops.repeat(
                            encoded_input["decoder_input_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                        inputs["decoder_attention_mask"] = einops.repeat(
                            encoded_input["decoder_attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    target_idx = (
                        target_label[i] if n_sampling_steps > 1 else target_label
                    )
                    if self.model_type == 't5':
                        assert probs.size(1) == len(target_idx)
                        target_probs = [probs[:, q, target_idx_] for q, target_idx_ in enumerate(target_idx)]

                        grad = torch.autograd.grad(
                            torch.unbind(torch.cat(target_probs, dim=0)), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)
                    # elif self.model_type == 'chatglm2':
                    #     grads = [torch.autograd.grad(torch.sum(prob), batch_weights)[0] for prob in torch.unbind(probs[:, target_idx])]
                    #     grad = torch.stack(grads).sum(dim=0)
                    #     integrated_grads_this_step.append(grad)
                    else:
                        grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_idx]), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                
                if self.model_type == "chatglm2":
                    baseline_activations = baseline_activations.mean(dim=0)
                    # baseline_activations = baseline_activations[1]
                    integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                else:
                    integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and ("qwen" in self.model_type or self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def modify_activations(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        all_layers = {n[0] for n in neurons}

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def suppress_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
        self,
        prompt: str,
        neurons: List[List[int]],
        target: str,
        mode: str = "edit",
        erase_value: str = "zero",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the
        # argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            # assert (
            #     self.model_type == "bert"
            # ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0] if len(argmax_tokens) == 1 else argmax_tokens
            if self.model_type == "gpt2" or self.model_type == "chatglm2":
                word_embeddings_weights = word_embeddings_weights.weight
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            # if len(target_label) > 1:
            #     target_label = target_label[0]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt2" or self.model_type=='chatglm2':
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt2" or self.model_type=='chatglm2':
                    if original_prediction_embedding.device != output_ff_weights.device:
                        original_prediction_embedding = original_prediction_embedding.to(output_ff_weights.device)
                    if target_embedding.device != output_ff_weights.device:
                        target_embedding = target_embedding.to(output_ff_weights.device)
                    if original_prediction_embedding.ndim > 1:
                        for oe in original_prediction_embedding:
                            output_ff_weights[position, :] -= oe
                    else:
                        output_ff_weights[position, :] -= original_prediction_embedding * 2
                    if target_embedding.ndim > 1:
                        for te in target_embedding:
                            output_ff_weights[position, :] += te
                    else:
                        output_ff_weights[position, :] += target_embedding * 2
                else:
                    if original_prediction_embedding.device != output_ff_weights.device:
                        original_prediction_embedding = original_prediction_embedding.to(output_ff_weights.device)
                    if target_embedding.device != output_ff_weights.device:
                        target_embedding = target_embedding.to(output_ff_weights.device)
                    if original_prediction_embedding.ndim > 1:
                        for oe in original_prediction_embedding:
                            output_ff_weights[:, position] -= oe
                    else:
                        output_ff_weights[:, position] -= original_prediction_embedding * 2
                    if target_embedding.ndim > 1:
                        for te in target_embedding:
                            output_ff_weights[:, position] += te
                    else:
                        output_ff_weights[:,position] += target_embedding * 2

            else:
                if self.model_type == "gpt2" or self.model_type=='chatglm2':
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt2" or self.model_type=='chatglm2':
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
        self,
        prompt: str,
        target: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
        self,
        prompt: str,
        neurons: List[List[int]],
        erase_value: str = "zero",
        target: Optional[str] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )
