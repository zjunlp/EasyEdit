import json
import os
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.evaluate import ALG_DICT, HPARAMS_DIR
from experiments.evaluate import main as eval_main

TMP_PARAMS_NAME = "sweep_params_tmp_{}_.json"


def main(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    sweep_key: str,
    sweep_vals: List,
    num_records: int,
    skip_generation_tests: bool,
    parallel_id: str,
):
    # Get current parameters
    with open(HPARAMS_DIR / alg_name / hparams_fname, "r") as f:
        data = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # Execute sweep
    tmp_params_path = HPARAMS_DIR / alg_name / TMP_PARAMS_NAME.format(parallel_id)
    for val in sweep_vals:
        data[sweep_key] = val
        with open(tmp_params_path, "w") as f:
            json.dump(data, f)

        eval_main(
            alg_name,
            model_name=(model, tok),
            hparams_fname=TMP_PARAMS_NAME.format(parallel_id),
            dataset_size_limit=num_records,
            continue_from_run=None,
            skip_generation_tests=skip_generation_tests,
            conserve_memory=False,
            dir_name=f"{alg_name}_{sweep_key}_sweep_{parallel_id}",
        )

    os.remove(tmp_params_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", choices=["ROME", "FT"], required=True)
    parser.add_argument(
        "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"], required=True
    )
    parser.add_argument("--hparams_fname", type=str, required=True)
    parser.add_argument("--sweep_key", type=str, required=True)
    parser.add_argument(
        "--sweep_vals", type=lambda x: list(map(float, x.split(","))), required=True
    )
    parser.add_argument("--num_records", type=int, default=50)
    parser.add_argument(
        "--use_generation_tests", dest="skip_generation_tests", action="store_false"
    )
    parser.set_defaults(skip_generation_tests=True)
    # Must be unique to prevent conflicts when simultaenously running multiple sweeps
    parser.add_argument("--parallel_id", type=str, required=True)

    args = parser.parse_args()

    assert (
        args.sweep_key in ALG_DICT[args.alg_name][0].KEYS
    ), f"sweep_key {args.sweep_key} not recognized"

    # Special case to handle layers
    if args.sweep_key == "layers":
        args.sweep_vals = [[int(x)] for x in args.sweep_vals]

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.sweep_key,
        args.sweep_vals,
        args.num_records,
        args.skip_generation_tests,
        args.parallel_id,
    )
