from tqdm import tqdm
from .deepedit_api_hparams import DeepEditApiHyperParams
from typing import Any, Dict, List, Tuple
from .utils import *
import zhipuai
def apply_deepedit_api_to_model(
        datasets,
        hparams: DeepEditApiHyperParams,
        **kwargs
    ):

    # Build a memory index which contains all the edits
    dataset = datasets
    new_facts = set()
    for d in dataset:
        for r in d["requested_rewrite"]:
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
    new_facts = list(new_facts)
    
    embs = get_sent_embeddings(new_facts, contriever, tokenizer)
    
    # run deepedit
    cor = 0
    tot = 0
    metrics = []
    for d in tqdm(dataset[0:100]):
        new_facts = set()
        
        for r in d["requested_rewrite"]:
            new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')
            
        fact_ls = list(new_facts)
        tot += 1
        for q in d["questions"][:1]:
            state_ls = []
            initial_state = {
                'last_entity' : 'Crouching Tiger',
                'new_entity' : newEntity('No entity appears.', q),
                'prompt' : task_prompt + "\n\nQustion: " + q + '\nThoughts with New Knowledge: ',
                'last_thought' : q,
                'count' : 1,
            }
            state_ls.append(initial_state)
            ans = None
            while len(state_ls) > 0:
                state_ = state_ls[-1]
                del state_ls[-1]
                if state_['count'] >= 9:
                    continue

                output = oneStep(state_['prompt'], state_['last_entity'], state_['new_entity'], state_['last_thought'], state_['count'], fact_ls)
                if isinstance(output, list):
                    state_ls.extend(output)
                else:
                    ans = output
                    break
            
            # if count overflow
            if ans is None:
                message = f'Ans is None! Multi-hop acc = {cor / tot} ({cor} / {tot})\n'
                print(message)
                metrics.append(message)
                continue

            # if the answer is correct
            if ans.lower() == d["new_answer"].lower() or ans.lower() in [ans_.lower() for ans_ in d["new_answer_alias"]]:
                cor += 1
                message = f'Multi-hop acc = {cor / tot} ({cor} / {tot})]\n'
                print(message)
                metrics.append(message)
                break
            else:
                message = f'Multi-hop acc = {cor / tot} ({cor} / {tot})\n'
                print(message)
                metrics.append(message)
                pass
    
    # print the final answer
    message = f'Multi-hop acc = {cor / tot} ({cor} / {tot})\n'
    print(message)
    metrics.append(message)
    
    return metrics