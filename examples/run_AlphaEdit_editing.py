import os.path
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.append('..')
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)


REMOTE_ROOT_URL = "https://memit.baulab.info/data/dsets"
URL_DICT = {
    "MCF" : f"{REMOTE_ROOT_URL}/multi_counterfact.json",
    "CF" : f"{REMOTE_ROOT_URL}/counterfact.json",
    "ZsRE" : f"{REMOTE_ROOT_URL}/zsre_mend_eval.json"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['ZsRE', 'MCF', 'CF'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--sequential_edit', action="store_true")

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    else:
        raise NotImplementedError

    url = URL_DICT[args.data_type]
    data_dir = Path(args.data_dir)

    if args.data_type == 'ZsRE':
        zsre_loc = data_dir / 'zsre_mend_eval.json'
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {url}")
            torch.hub.download_url_to_file(url, zsre_loc)
        with open(zsre_loc, "r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'CF':
        cf_loc = data_dir / 'counterfact.json'
        if not cf_loc.exists():
            print(f"{cf_loc} does not exist. Downloading from {url}")
            torch.hub.download_url_to_file(url, cf_loc)
        with open(cf_loc,"r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['requested_rewrite']['prompt'].format(edit_data_['requested_rewrite']['subject']) for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['paraphrase_prompts'][0] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['target_new']['str'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['neighborhood_prompts'] for edit_data_ in edit_data]
        locality_ans = [[edit_data_['requested_rewrite']["target_true"]["str"]] * len(edit_data_["neighborhood_prompts"]) for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'MCF':
        mcf_loc = data_dir / 'multi_counterfact.json'
        if not mcf_loc.exists():
            print(f"{mcf_loc} does not exist. Downloading from {url}")
            torch.hub.download_url_to_file(url, mcf_loc)
        with open(mcf_loc,"r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['requested_rewrite']['prompt'].format(edit_data_['requested_rewrite']['subject']) for edit_data_ in edit_data]
        subject = [edit_data_['requested_rewrite']['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['paraphrase_prompts'][0] for edit_data_ in edit_data]
        target_new = [edit_data_['requested_rewrite']['target_new']['str'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['neighborhood_prompts'] for edit_data_ in edit_data]
        locality_ans = [[edit_data_['requested_rewrite']["target_true"]["str"]] * len(edit_data_["neighborhood_prompts"]) for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        keep_original_weight=True,
        sequential_edit=args.sequential_edit,
        test_generation=True,
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

