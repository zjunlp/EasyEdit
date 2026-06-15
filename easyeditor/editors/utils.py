from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np
import random
import math
import torch

from ..util import nethook
from ..util.device import copy_to_param


CALLABLE_RESTORE_ALGS = {"KN", "GRACE", "WISE"}
PEFT_RESTORE_ALGS = {"LoRA", "QLoRA", "DPO"}
KEEP_EDITED_MODEL_ALGS = {"MELO"}


def restore_after_edit(editor, edited_model, weights_copy):
    """Restore editor.model after a non-sequential edit."""
    alg_name = getattr(editor, "alg_name", None)

    if alg_name in CALLABLE_RESTORE_ALGS:
        if weights_copy is not None:
            with torch.no_grad():
                weights_copy()
        return edited_model

    if alg_name in PEFT_RESTORE_ALGS:
        restored_model = edited_model.unload()
        if restored_model is not None:
            editor.model = restored_model
        if hasattr(editor.model, "peft_config"):
            del editor.model.peft_config
        return edited_model

    if alg_name in KEEP_EDITED_MODEL_ALGS:
        editor.model = edited_model
        return edited_model

    if weights_copy is not None:
        for key, value in weights_copy.items():
            copy_to_param(nethook.get_parameter(editor.model, key), value)

    return edited_model


def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]
        
def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys

def get_section_acc_keys(dict_list, phase, section):
    all_keys = set()
    for metric in dict_list:
        phase_metrics = metric.get(phase, {})
        section_metrics = phase_metrics.get(section, {})
        if isinstance(section_metrics, dict):
            for key in section_metrics.keys():
                if key.endswith("acc"):
                    all_keys.add(key)
    return all_keys

def collect_metric_values(dict_list, phase, key):
    values = []
    for metric in dict_list:
        phase_metrics = metric.get(phase, {})
        if key in phase_metrics:
            values.append(phase_metrics[key])
    return values

def get_metric_protocol_summary(dict_list):
    protocol_summary = {}
    for metric in dict_list:
        for phase in ["pre", "post"]:
            phase_metrics = metric.get(phase, {})
            metric_meta = phase_metrics.get("metric_meta", {})
            for section, meta in metric_meta.items():
                key = f"{phase}.{section}"
                comparable_group = meta.get("comparable_group", "unknown")
                protocol_summary.setdefault(key, set()).add(comparable_group)

    return {
        key: sorted(groups)
        for key, groups in sorted(protocol_summary.items())
    }
    
def summary_metrics(all_metrics):
    if isinstance(all_metrics, dict):
        all_metrics = [all_metrics, ]
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    output_file = os.path.join(logs_dir, 'results.json')
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    mean_metrics = dict()
    for eval in ["pre", "post"]:
        mean_metrics[eval] = dict()
        for key in ["rewrite_acc", "rephrase_acc", 'rewrite_ppl', 'ood_acc']:
            values = collect_metric_values(all_metrics, eval, key)
            if len(values) > 0:
                mean_metrics[eval][key] = np.mean(values)
        for section in ["locality", "portability"]:
            section_keys = get_section_acc_keys(all_metrics, eval, section)
            if len(section_keys) > 0:
                mean_metrics[eval][section] = dict()
                for lkey in section_keys:
                    metrics = []
                    for metric in all_metrics:
                        section_metrics = metric.get(eval, {}).get(section, {})
                        if isinstance(section_metrics, dict) and lkey in section_metrics:
                            metrics.append(np.mean(section_metrics[lkey]))
                    if len(metrics) > 0:
                        mean_metrics[eval][section][lkey] = np.mean(metrics)
    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
    protocol_summary = get_metric_protocol_summary(all_metrics)
    if protocol_summary:
        mean_metrics["metric_protocols"] = protocol_summary
        print("Metric Protocol Summary: ", protocol_summary)
        for metric_key, comparable_groups in protocol_summary.items():
            if len(comparable_groups) > 1:
                print(
                    f"WARNING: {metric_key} contains multiple comparable_group values: "
                    f"{comparable_groups}. Do not average or compare these scores directly."
                )

    print("Metrics Summary: ", mean_metrics)

def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Union[str, List[str]],
                      target_neg: Optional[Union[str, List[str]]] = None,
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_,
        'portability': {},
        'locality': {}
    }
    for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
    ]

    if target_neg is not None:
        if isinstance(target_neg, str):
            target_neg = [target_neg,]
        assert len(target_neg) == len(prompts)
        for i, request in enumerate(requests):
            request.update(
                {
                    'target_neg': target_neg[i]
                }
            )

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        if len(kwargs['loc_prompts']) < len(requests):
            kwargs['loc_prompts'] = (kwargs['loc_prompts'] * math.ceil(len(requests) / len(kwargs['loc_prompts'])))[:len(requests)]
            random.shuffle(kwargs['loc_prompts'])
        assert len(kwargs['loc_prompts']) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts,]

        for i, request in enumerate(requests):
            request.update(
                {
                    'rephrase_prompt': rephrase_prompts[i],
                }
            )
    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            == len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            == len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
    return requests
