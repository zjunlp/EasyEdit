import os.path
# sys.path.append('..')
import numpy as np
from easyeditor import (
    DINMHyperParams,
    )
from easyeditor import SafetyEditor
from easyeditor import DINMHyperParams
from easyeditor import SafetyDataset
import json
from easyeditor import n_gram_entropy

import argparse



def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--NLPCC', help='execute the training process for Track 2 of NLPCC', default=True)
    parser.add_argument('--editing_method', default="DINM", type=str)  
    parser.add_argument('--hparams_dir', required=True, type=str)  
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--results_save_dir', required=True, type=str)

    args = parser.parse_args()

    # this file is only for DINM
    if args.editing_method == 'DINM':
        editing_hparams = DINMHyperParams
    else:
        raise NotImplementedError
    output_dir = f'{args.results_save_dir}/detoxify_train.json'
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = SafetyEditor.from_hparams(hparams)

    # we take an example from "three_instances_for_editing.json". You can choose a case that fits your scenario to edit the vanilla model.
    edit_data_all = SafetyDataset(f'{args.data_dir}/three_instances_for_editing.json')
    edit_data = [edit_data_all[1],]

    case_id = [edit_data_['case_id'] for edit_data_ in edit_data]
    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    prompts_with_systemPrompt = [edit_data_['prompt'] + ' ' + hparams.suffix_system_prompt for edit_data_ in edit_data]
    target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
    ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
    locality_prompts_with_systemPrompt = [edit_data_['locality_prompt'] + ' ' + hparams.suffix_system_prompt for edit_data_ in edit_data]
    locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
    locality_inputs = {
        'general knowledge constraint': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    locality_inputs_with_systemPrompt = {
        'general knowledge constraint': {
            'prompt': locality_prompts_with_systemPrompt,
            'ground_truth': locality_ans
        },
    }

    # Execute the DINM method and save the edited LLM to the ckpt_save_dir path.
    editor.edit(
        NLPCC = args.NLPCC,
        ckpt_save_dir = f'{args.results_save_dir}/dinm_llama2-chat',
        prompts=prompts,
        prompts_with_systemPrompt = prompts_with_systemPrompt,
        target_new=target_new,
        ground_truth=ground_truth,
        locality_inputs=locality_inputs,
        locality_inputs_with_systemPrompt = locality_inputs_with_systemPrompt,
        keep_original_weight=True,
    )
    
    print("Training is done")

# Then you can execute test_detoxify_generate.py via the edited model (the path is ckpt_save_dir) to evaluate the detoxifing performance and obtain detoxify_val.json
# Besides, you should also assess the side effects, i.e., the general performance of edited model for unharmfullless user query. You can leverage opencompass to achive general_val.json
        

# DINM edits llama-2-7b-chat
# python train_DINM_for_NLPCC.py --hparams_dir ./hparams/DINM/llama-7b --results_save_dir ./safety_results
    








    
