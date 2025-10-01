import os.path
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.append('..')
from easyeditor import (
    SimIEHyperParams,
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
                        choices=['ZsRE', 'counterfact', 'hallucination'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=1000, type=int)
    parser.add_argument('--sequential_edit', action="store_true")

    args = parser.parse_args()

    if args.editing_method == 'SimIE':
        editing_hparams = SimIEHyperParams
    else:
        raise NotImplementedError

    url = URL_DICT[args.data_type]
    data_dir = Path(args.data_dir)
    K = args.ds_size

    if args.data_type == 'ZsRE':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

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
    elif args.data_type == 'counterfact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.base_hparams.model_name.split("/")[-1]}_SimIE_{hparams.base_method}_'
        f'{args.data_type}_N={args.ds_size}_'
        f'lamHyper={hparams.lamHyper}_'
        f'solver={hparams.solver}_'
        f'init={hparams.init_model}.json'
    )

    print("See results at: ", output_file)

    editor = BaseEditor.from_hparams(hparams)
    # For SimIE, sequential_edit must be True
    if not args.sequential_edit:
        print("[Warning] SimIE requires sequential_edit=True. Setting it automatically.")
        args.sequential_edit = True
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

