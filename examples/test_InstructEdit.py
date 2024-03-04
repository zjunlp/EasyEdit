import os.path
import sys
sys.path.append('..')
sys.path.append('../..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from easyeditor.models.lora import LoRAHyperParams
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset, WikiRecentDataset

import argparse
random.seed(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, choices=['convsent', 'counterfact', 'wikirecent', 'zsre'])
    parser.add_argument('--data', default=None, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--inst_index', default=None, type=str, choices=[None, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND' or args.editing_method.lower() == 'instructedit':
        editing_hparams = MENDHyperParams
    else:
        raise NotImplementedError

    test_data = json.load(open(os.path.join(args.data_dir, args.data), 'r', encoding='utf-8'))

    if args.ds_size is not None:
        test_data = test_data[:args.ds_size]

    temp = "Task: {}\nDescription: {}\nInput: {}"
    desc = {
        "convsent": 
            [
                "Teach the chatbot to sound [LABEL] when talking about [TOPIC], but keep its cool on everything else.",
                "Get the chatbot to show a [LABEL] mood only when [TOPIC] comes up, not messing with other stuff.",
                "Help the chatbot pick up a [LABEL] tone on [TOPIC], and not change its tune on other matters.",
                "Make sure the chatbot gives off a [LABEL] feel when it chats about [TOPIC], without going off-key on other topics.",
                "Have the chatbot throw in a [LABEL] sentiment when it gets to [TOPIC], leaving its opinion on other things unchanged.",
                "Guide the chatbot to lean [LABEL] when the convo hits [TOPIC], but stay neutral when it's not about that.",
                "Set the chatbot to hit a [LABEL] note when [TOPIC] is in the spotlight, without shifting its mood for other chats.",
                "Train the chatbot to be [LABEL] about [TOPIC], and not let that affect its chit-chat on different things.",
                "Fix the chatbot's reaction to be [LABEL] when it's about [TOPIC], but not tinker with its other topic reactions.",
                "Steer the chatbot towards a [LABEL] attitude about [TOPIC], but make sure it doesn't sway its stance elsewhere.", ## The last one for testing instruction generality.
            ],
        "counterfact": 
            [
                "A dataset designed to challenge and assess models on their ability to capture often overlooked tail entities.",
                "A test set for measuring how well models can identify and deal with less common or 'tail' entities.",
                "A benchmarking tool that helps evaluate the effectiveness of model editing methods in recognizing rare entities.",
                "A dataset that provides a critical look at how well models can edit and update their methods to include tail entities.",
                "An evaluation dataset focused on the model's ability to handle entities that are often missed in predictions.",
                "A dataset that provides a way to test the robustness of models against the challenge of detecting tail entities.",
                "A specialized dataset for gauging the performance of models in identifying entities typically neglected in data processing.",
                "A testbed for analyzing the adaptability of models to identify and incorporate frequently missed tail entities.",
                "An assessment dataset that targets the weak spots of models in detecting and incorporating tail entities.",
                "A dataset curated to push the boundaries of model's capabilities in recognizing and processing tail entities.",
                ],
        "wikirecent": 
            [
                "A curated collection of the latest factual relationships added to WikiData.",
                "An up-to-date dataset for keeping models informed with the newest WikiData entries.",
                "A dynamic repository capturing the newest edits and additions to WikiData entities.",
                "A dataset designed to reflect the latest knowledge graph updates on WikiData.",
                "A continuous feed of WikiData's latest verified triplets for data enrichment.",
                "A specialized dataset aimed at integrating recent WikiData updates into models.",
                "A streamlined dataset offering the most recent WikiData additions for machine learning.",
                "A contemporary dataset serving the latest WikiData contributions for real-time updating.",
                "A regularly updated dataset that captures the evolving landscape of WikiData's knowledge graph.",
                "A dataset focusing on the integration of newly verified factual data from WikiData.",
                ],
          "zsre": 
              [
                  "A dataset aimed at answering questions without context, focusing solely on the relationship between subjects and objects.",
                  "A collection for developing AI that can deduce correct objects based on given subjects and their relations.",
                  "A question-answering resource that challenges models to identify objects from specified subjects and relations.",
                  "A dataset designed to test a model's ability to connect subjects and relations to their rightful objects.",
                  "An evaluation tool for assessing how well a model can infer objects from a given subject-relation pair.",
                  "A benchmark dataset for validating the accuracy of models in providing objects for stated subjects and relations.",
                  "A dataset facilitating the assessment of models' capacity to answer questions based on subject-relation prompts.",
                  "A tool for measuring a model's proficiency in identifying objects based on their relationship with a subject.",
                  "A dataset tailored for training models to autonomously find correct objects from given subjects and relations.",
                  "A dataset for driving the development of AI that can predict objects given a subject and its relation.",
              ]
        }
    
    if args.inst_index is not None:
        description = desc[args.data_type][int(args.inst_index)]
    else:
        description = random.choice(desc[args.data_type])
    template = temp.format(args.data_type, description, "{}")
    if "prompt" in test_data[0].keys():
        if args.editing_method.lower() == 'instructedit':
            prompts = [template.format(test_data_['prompt']) for test_data_ in test_data]
            rephrase_prompts = [template.format(edit_data_['rephrase']) \
                                if 'rephrase' in edit_data_.keys() else template.format(edit_data_['prompt']) for edit_data_ in test_data] 
        else:
            prompts = [test_data_['prompt'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase'] if 'rephrase' in edit_data_.keys() else edit_data_['prompt'] for edit_data_ in test_data]  
        
        target_new = [edit_data_['target_new'] for edit_data_ in test_data]
        locality_inputs = [edit_data_['locality'] for edit_data_ in test_data]
        portability_inputs = [edit_data_['portability'] for edit_data_ in test_data]
        subject = [edit_data_['subject'] for edit_data_ in test_data]
    elif "src" in test_data[0].keys():
        if args.editing_method.lower() == 'instructedit':
            prompts = [template.format(test_data_['src']) for test_data_ in test_data]
            rephrase_prompts = [template.format(edit_data_['rephrase']) for edit_data_ in test_data]  
            portability_inputs = [{'prompt': template.format(edit_data_['portability']['New Question']), 'ground_truth': edit_data_['portability']['New Answer']} for edit_data_ in test_data]
        else:
            prompts = [test_data_['src'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data] 
            portability_inputs = [{'prompt': edit_data_['portability']['New Question'], 'ground_truth': edit_data_['portability']['New Answer']} for edit_data_ in test_data] 
        
        target_new = [edit_data_['alt'] for edit_data_ in test_data]
        locality_inputs = [{'prompt': edit_data_['loc'], 'ground_truth': edit_data_['loc_ans']} for edit_data_ in test_data]
        subject = [edit_data_['subject'] for edit_data_ in test_data]        
    
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': target_new_,
            'subject': subject_,
            'rephrase_prompt': rephrase_,
            'portability': {},
            'locality': {}
        }
    for prompt, subject_, target_new_, rephrase_ in zip(prompts, subject, target_new, rephrase_prompts)
    ]
    print(len(requests))
            
    if "prompt" in test_data[0].keys():
        for i, request in enumerate(requests):
            locality_item = locality_inputs[i]
            for locality_key in locality_item.keys():
                prompts = [x['prompt'] for x in locality_item[locality_key]]
                ground_truth = [x['ground_truth'][0][0] for x in locality_item[locality_key]]
                request['locality'].update(
                    {
                        locality_key: {
                            f'prompt': prompts,
                            f'ground_truth': ground_truth
                        }
                    }
                )
            portability_item = portability_inputs[i]
            for portability_key in portability_item.keys():
                if args.editing_method.lower() == 'instructedit':
                    prompts = [template.format(x['prompt']) for x in portability_item[portability_key]]
                else:
                    prompts = [x['prompt'] for x in portability_item[portability_key]]
                ground_truth = [x['ground_truth'][0][0] for x in portability_item[portability_key]]
                request['portability'].update(
                    {
                        portability_key: {
                            f'prompt': prompts,
                            f'ground_truth': ground_truth
                        }
                    }
                )
    elif "src" in test_data[0].keys():
        for i, request in enumerate(requests):
            locality_item = locality_inputs[i]
            prompts = locality_item['prompt']
            ground_truth = locality_item['ground_truth']
            request['locality'].update(
                {
                    "locality": {
                        f'prompt': prompts,
                        f'ground_truth': ground_truth
                    }
                }
            )
            portability_item = portability_inputs[i]
            prompts = portability_item['prompt']
            ground_truth = portability_item['ground_truth']
            request['portability'].update(
                {
                    "portability": {
                        f'prompt': prompts,
                        f'ground_truth': ground_truth
                    }
                }
            )    
    print(requests[0])
    editor = BaseEditor.from_hparams(hparams)
    print("begin editing")
    metrics, edited_model, _ = editor.edit_requests(
        requests=requests,
        keep_original_weight=True,
    )
    try:
        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir)
    except Exception as e:
            print(f"Failed to create directory: {e}")
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, 
                                         f'{args.editing_method}_{args.data_type}{(("_inst_id_" + args.inst_index) if args.inst_index is not None else "")}_results.json'), 
                            'w'), indent=4)