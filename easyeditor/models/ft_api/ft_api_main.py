import copy
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import time

import openai

from .ft_api_hparams import FTApiHyperParams


def apply_ft_api_to_model(
        requests: List[Dict],
        hparams: FTApiHyperParams,
        keep_original_weight=False,
        **kwargs
    ):

    if len(requests) < 10:
        extend_requests = copy.deepcopy(requests)

        while(len(extend_requests) < 10):
            extend_requests.extend(requests)
        extend_requests = extend_requests[:10]

        print(f"Original length: {len(requests)}.\n FT-Api requires at least 10 samples, we have copied your sample several times",
              f"and the current sample length is {len(extend_requests)}.")
    else:
        extend_requests = copy.deepcopy(requests)
        print(f'The current sample length is {len(extend_requests)}.')

    for request in requests:
        print(
            f"Executing FT-Api algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    example_dir = os.path.join(hparams.results_dir, 'FT-Api', 'example.jsonl')
    os.makedirs(os.path.join(hparams.results_dir, 'FT-Api'), exist_ok=True)

    openai.api_key = hparams.api_key

    if hparams.proxy is not None:
        openai.proxy = hparams.proxy

    with open(example_dir, 'w', encoding='utf-8') as fout:
        for request in extend_requests:
            temp_dict = {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
                          {"role": "user", "content": f"{request['prompt']}"},
                          {"role": "assistant", "content": f"{request['target_new']}"}]}
            json_str = json.dumps(temp_dict)
            fout.write(json_str)
            fout.write('\n')

    openai_file = openai.File.create(
        file=open(example_dir, "rb"),
        purpose='fine-tune'
    )

    print(openai_file)

    # wait file uploading
    while(openai.File.retrieve(f"{openai_file['id']}")['status'] == 'uploaded'):
        pass

    openai_job = openai.FineTuningJob.create(training_file=f"{openai_file['id']}",
                                             model=f"{hparams.model_name}",
                                             n_epochs=hparams.n_epochs)

    start = time.time()
    while True:
        edited_model = openai.FineTuningJob.retrieve(f"{openai_job['id']}")['fine_tuned_model']

        if edited_model is None:
            print(f'Waiting for openai to complete the fine-tuning task!!! Time Cost:{time.time() - start}s.')
            time.sleep(10)
        else:
            break
    print(f'\nfine-tuning task done...., finetuned model name is {edited_model}')

    return edited_model, hparams.model_name

