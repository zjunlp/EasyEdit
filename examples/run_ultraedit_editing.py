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
    UltraEditHyperParams,
    BaseEditor,
    summary_metrics,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['wikibigedit', 'ultraeditbench', 'zsre'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
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
    elif args.editing_method == 'UltraEdit':
        editing_hparams = UltraEditHyperParams
    else:
        raise NotImplementedError

    data_dir = Path(args.data_dir)

    if args.data_type == 'zsre':
        zsre_dir = data_dir / 'zsre_eval_20k.json'
        with open(zsre_dir, "r") as f:
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
    elif args.data_type == 'wikibigedit':
        wiki_dir = data_dir / 'wikibigedit_eval_17k.json'
        with open(wiki_dir, "r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['update'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['ans'] for edit_data_ in edit_data]
        portability_personas_prompts = [[data['personas']] if isinstance(data['personas'], str) else None for data in edit_data]
        portability_personas_answers = [[data['ans']] for data in edit_data]
        portability_hop_prompts = [[data['mhop']] if isinstance(data['mhop'], str) else None for data in edit_data]
        portability_hop_answers = [[data['mhop_ans']] if isinstance(data['mhop_ans'], str) else None for data in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]

        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

        portability_inputs = {
            'personas':{ 
                'prompt': portability_personas_prompts,
                'ground_truth': portability_personas_answers  
            },
            'mhop':{
                'prompt': portability_hop_prompts,
                'ground_truth': portability_hop_answers
            }
        }

    elif args.data_type == 'ultraeditbench':
        ultraeditbench_dir = data_dir / 'UltraEditBench_2M.json'
        with open(ultraeditbench_dir,"r") as f:
            raw = json.load(f)
        edit_data = raw[:args.ds_size]
        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
        target_new = [edit_data_['ans'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
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
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs if args.data_type == 'wikibigedit' else None,
        keep_original_weight=True,
        sequential_edit=args.sequential_edit,
        test_generation=True,
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

