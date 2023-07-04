# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`

import argparse
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import torch
from knowledge_neurons import (
    ALL_MODELS,
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
    pararel_expanded,
)
from tqdm import tqdm

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--local_rank", help="local rank for multigpu processing", type=int
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help=f"name of the LM to use - choose from {ALL_MODELS}",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="bert_base_uncased_neurons",
        help="directory in which to save results",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of steps to run the integrated gradient calculation for",
    )
    parser.add_argument(
        "--adaptive_threshold",
        type=int,
        default=0.3,
        help="A setting used to determine the score threshold above which coarse neurons are selected - the paper uses 0.3",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="the threshold for the sharing percentage - we retain neurons that are shared by p% of prompts (p here is a decimal fraction, i.e between 0 and 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    RESULTS_DIR = Path(args.results_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)

    # load dataset
    # each item in pararel is the same 'fact' (head/relation/tail) expressed in different ways
    PARAREL = pararel_expanded()

    ##############################################################################
    # data parallel stuff
    NUM_REPLICAS = torch.cuda.device_count()
    INDICES = list(range(len(PARAREL)))
    INDICES = INDICES[args.local_rank : len(PARAREL) : NUM_REPLICAS]
    KEYS = list(PARAREL.keys())
    torch.cuda.set_device(args.local_rank)
    ##############################################################################

    # initialize results dicts
    RESULTS = {}
    NEURONS = {}

    # setup model + tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(args.model_name))

    # because we may end up getting some neurons multiple times, use lru cache to save time
    @lru_cache(maxsize=None)
    def get_neurons(_uuid):
        PROMPTS, GROUND_TRUTH, RELATION_NAME = (
            PARAREL[_uuid]["sentences"],
            PARAREL[_uuid]["obj_label"],
            PARAREL[_uuid]["relation_name"],
        )
        neurons = kn.get_refined_neurons(
            prompts=PROMPTS,
            ground_truth=GROUND_TRUTH.lower(),
            p=args.p,
            batch_size=args.batch_size,
            steps=args.steps,
            coarse_adaptive_threshold=args.adaptive_threshold,
            quiet=True,
        )
        return neurons, PARAREL[_uuid]

    def get_unrelated_fact(KEYS, uuid):
        n_keys = len(KEYS)
        while True:
            random_uuid = KEYS[random.randint(0, n_keys - 1)]
            if random_uuid == uuid:
                continue
            return random_uuid

    # go through each item in the PARAREL dataset, get the refined neurons, save them, and evaluate the results when suppressing the
    # refined neurons vs. unrelated neurons.
    for i, idx in enumerate(tqdm(INDICES, position=args.local_rank)):
        uuid = KEYS[idx]
        neurons, data = get_neurons(uuid)  # get refined neurons
        unrelated_uuid = get_unrelated_fact(
            KEYS, uuid
        )  # get a uuid for an unrelated fact / relation
        unrelated_neurons, unrelated_data = get_neurons(
            unrelated_uuid
        )  # get the unrelated neurons

        # initialize a results dict
        results_this_uuid = {
            "suppression": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(unrelated_data["sentences"]),
                },
            },
            "enhancement": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(unrelated_data["sentences"]),
                },
            },
        }

        for PROMPT in data["sentences"]:
            gt = data["obj_label"].lower()
            # really should be using a different for the suppression, but the authors didn't make their bing dataset available
            suppression_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )
            enhancement_results, _ = kn.enhance_knowledge(
                PROMPT, gt, neurons, quiet=True
            )

            # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
            suppression_prob_diff = (
                suppression_results["after"]["gt_prob"]
                - suppression_results["before"]["gt_prob"]
            ) / suppression_results["before"]["gt_prob"]
            results_this_uuid["suppression"]["related"]["pct_change"].append(
                suppression_prob_diff
            )

            enhancement_prob_diff = (
                enhancement_results["after"]["gt_prob"]
                - enhancement_results["before"]["gt_prob"]
            ) / enhancement_results["before"]["gt_prob"]
            results_this_uuid["enhancement"]["related"]["pct_change"].append(
                enhancement_prob_diff
            )

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["related"]["correct_before"].append(
                suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["related"]["correct_after"].append(
                suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["related"]["correct_before"].append(
                enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["related"]["correct_after"].append(
                enhancement_results["after"]["argmax_completion"] == gt
            )

        for PROMPT in unrelated_data["sentences"]:
            # do the same but with unrelated facts

            gt = unrelated_data["obj_label"].lower()

            unrelated_suppression_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )
            unrelated_enhancement_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )

            # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
            suppression_prob_diff = (
                unrelated_suppression_results["after"]["gt_prob"]
                - unrelated_suppression_results["before"]["gt_prob"]
            ) / unrelated_suppression_results["before"]["gt_prob"]
            results_this_uuid["suppression"]["unrelated"]["pct_change"].append(
                suppression_prob_diff
            )
            enhancement_prob_diff = (
                unrelated_enhancement_results["after"]["gt_prob"]
                - unrelated_enhancement_results["before"]["gt_prob"]
            ) / unrelated_enhancement_results["before"]["gt_prob"]
            results_this_uuid["enhancement"]["unrelated"]["pct_change"].append(
                enhancement_prob_diff
            )

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["unrelated"]["correct_before"].append(
                unrelated_suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["unrelated"]["correct_after"].append(
                unrelated_suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["unrelated"]["correct_before"].append(
                unrelated_enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["unrelated"]["correct_after"].append(
                unrelated_enhancement_results["after"]["argmax_completion"] == gt
            )

        results_this_uuid["n_refined_neurons"] = len(neurons)
        results_this_uuid["n_unrelated_neurons"] = len(unrelated_neurons)
        results_this_uuid["relation_name"] = data["relation_name"]
        RESULTS[uuid] = results_this_uuid
        NEURONS[uuid] = neurons

    # save results + neurons to json file
    with open(
        RESULTS_DIR / f"{args.model_name}_pararel_neurons_{args.local_rank}.json", "w"
    ) as f:
        json.dump(NEURONS, f, indent=4)
    with open(
        RESULTS_DIR / f"{args.model_name}_pararel_results_{args.local_rank}.json", "w"
    ) as f:
        json.dump(RESULTS, f, indent=4)
