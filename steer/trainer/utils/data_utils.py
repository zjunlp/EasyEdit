from dataclasses import dataclass
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import set_seed
import transformers, datasets, torch
from typing import Dict, Optional, Sequence, Union, List, Any


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


@dataclass
class InterventionDataCollator(object):
    """Collate examples for Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator
    concept_tokenizer : transformers.AutoTokenizer = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        if self.concept_tokenizer is None:
            self.concept_tokenizer = self.tokenizer
            
                
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        max_concept_seq_len = max([len(inst["concept_input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])
            concept_non_pad_len = len(inst["concept_input_ids"])
            
            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()
            inst["prompt_intervention_masks"] = inst["intervention_masks"].clone()
            inst["prompt_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            _concept_input_id_paddings = torch.tensor(
                [self.concept_tokenizer.pad_token_id for _ in range(max_concept_seq_len - concept_non_pad_len)])
            inst["concept_input_ids"] = torch.cat((_concept_input_id_paddings, inst["concept_input_ids"])).int()
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()
            inst["concept_attention_mask"] = (inst["concept_input_ids"] != self.concept_tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs


@dataclass
class PreferenceInterventionDataCollator(object):

    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator
    preference_pairs: List[str]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # get max lengths for padding
        max_intervention_len = -1
        max_seq_len = -1
        for instance in instances:
            for k, v in instance.items():
                if "_intervention_locations" in k:
                    max_intervention_len = max(max_intervention_len, len(v[0]))
                if "_input_ids" in k:
                    max_seq_len = max(max_seq_len, len(v))

        for inst in instances:
            for pair in self.preference_pairs:
                winning_non_pad_len = len(inst[f"{pair}_winning_input_ids"])
                losing_non_pad_len = len(inst[f"{pair}_losing_input_ids"])

                # intervention locations
                _winning_intervention_location_paddings = torch.tensor(
                    [[winning_non_pad_len for _ in range(max_intervention_len - len(inst[f"{pair}_winning_intervention_locations"][0]))]])
                _losing_intervention_location_paddings = torch.tensor(
                    [[losing_non_pad_len for _ in range(max_intervention_len - len(inst[f"{pair}_losing_intervention_locations"][0]))]])
                inst[f"{pair}_winning_intervention_locations"] = torch.cat(
                    [inst[f"{pair}_winning_intervention_locations"], _winning_intervention_location_paddings], dim=-1).int()
                inst[f"{pair}_losing_intervention_locations"] = torch.cat(
                    [inst[f"{pair}_losing_intervention_locations"], _losing_intervention_location_paddings], dim=-1).int()
        
                # input ids
                _winning_input_id_paddings = torch.tensor(
                    [self.tokenizer.pad_token_id for _ in range(max_seq_len - winning_non_pad_len)])
                _losing_input_id_paddings = torch.tensor(
                    [self.tokenizer.pad_token_id for _ in range(max_seq_len - losing_non_pad_len)])
                inst[f"{pair}_winning_input_ids"] = torch.cat(
                    (inst[f"{pair}_winning_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _winning_input_id_paddings)).int()
                inst[f"{pair}_losing_input_ids"] = torch.cat(
                    (inst[f"{pair}_losing_input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _losing_input_id_paddings)).int()  
                
                # labels
                _winning_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - winning_non_pad_len+1)])
                _losing_label_paddings = torch.tensor([-100 for _ in range(max_seq_len - losing_non_pad_len+1)])
                inst[f"{pair}_winning_labels"] = torch.cat((inst[f"{pair}_winning_labels"], _winning_label_paddings))
                inst[f"{pair}_losing_labels"] = torch.cat((inst[f"{pair}_losing_labels"], _losing_label_paddings))
        
                # attention mask
                inst[f"{pair}_winning_attention_mask"] = (inst[f"{pair}_winning_input_ids"] != self.tokenizer.pad_token_id).int()
                inst[f"{pair}_losing_attention_mask"] = (inst[f"{pair}_losing_input_ids"] != self.tokenizer.pad_token_id).int()
            
        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, dataset, 
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    concept_tokenizer=None, 
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
        
    if concept_tokenizer is None:
        concept_tokenizer = tokenizer
    
    all_base_input_ids, all_intervention_locations, all_output_ids,  = [], [], []
    all_prompt_lengths = []
    
    all_concept_ids, all_concept_input_ids = [], []
    
    for item in dataset:
        
        _concept, _input, _output = item["output_concept"], item["input"], item["output"]
        
        all_concept_ids.append(item["concept_id"])
        
        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100
        
        concept_input_ids = concept_tokenizer(
            _concept, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]

        if positions is None or positions == "all_prompt":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
        elif positions == "all":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_length)]])
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length - prefix_length, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by prefix length
            shifted_intervention_locations = [[loc + prefix_length for loc in intervention_locations[0]]]
            intervention_locations = shifted_intervention_locations
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
        
        all_concept_input_ids.append(concept_input_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "prompt_lengths": all_prompt_lengths,
        "concept_input_ids": all_concept_input_ids,
        "concept_ids": torch.tensor(all_concept_ids)
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 'prompt_lengths', 'labels', 'concept_input_ids', 'concept_ids'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = InterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn, concept_tokenizer=concept_tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_prompt_suffix_length(tokenizer):
    message_a = [{"role": "user", "content": 'a'}]
    message_b = [{"role": "user", "content": 'b'}]
    tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True, add_generation_prompt=True)
    tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True, add_generation_prompt=True)
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])


def preprocess_preference_data(
    tokenizer, prompt, winning_output, losing_output, positions, prefix_length, prefix_tuning
):
    """For each condition above, we need to preprocess the data."""
    prompt_suffix_length, prompt_suffix = get_prompt_suffix_length(tokenizer)
    
    if isinstance(winning_output, float):
        winning_output = tokenizer.eos_token
    if isinstance(losing_output, float):
        losing_output = tokenizer.eos_token

    winning_input = prompt + winning_output
    losing_input = prompt + losing_output

    prompt_ids = tokenizer(
        prompt, 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt")["input_ids"][0]
    winning_input_ids = tokenizer(
        winning_input, 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt")["input_ids"][0]
    losing_input_ids = tokenizer(
        losing_input, 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt")["input_ids"][0]
    
    prompt_length = len(prompt_ids)
    winning_output_length = len(winning_input_ids)
    losing_output_length = len(losing_input_ids)

    # output ids with prompt token mask
    winning_output_ids = winning_input_ids.clone()
    losing_output_ids = losing_input_ids.clone()
    winning_output_ids[:prompt_length] = -100
    losing_output_ids[:prompt_length] = -100

    if prefix_tuning:
        # before trying this, we already tried adding a token before BOS, etc.. it did not work.
        winning_intervention_locations = torch.tensor([[prefix_length]]) # only intervene on the first token
        losing_intervention_locations = torch.tensor([[prefix_length]]) # only intervene on the first token
    elif positions is None or positions == "all_prompt":
        winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, prompt_length)]])
        losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, prompt_length)]])
    elif positions == "all":
        winning_intervention_locations = torch.tensor([[i for i in range(prefix_length, winning_output_length)]])
        losing_intervention_locations = torch.tensor([[i for i in range(prefix_length, losing_output_length)]])
    elif positions == "all_generation":
        winning_intervention_locations = torch.tensor([[i for i in range(
            prompt_length - prompt_suffix_length, winning_output_length)]])
        losing_intervention_locations = torch.tensor([[i for i in range(
            prompt_length - prompt_suffix_length, losing_output_length)]])
    elif "f" in positions or "l" in positions or "+" in positions:
        first_n, last_n = parse_positions(positions)
        intervention_locations = get_intervention_locations(
            last_position=prompt_length - prefix_length, 
            first_n=first_n, 
            last_n=last_n,
            pad_mode="last",
            num_interventions=1,
            share_weights=True,
        )
        # shift intervention locations by prefix length
        shifted_intervention_locations = [[loc + prefix_length for loc in intervention_locations[0]]]
        winning_intervention_locations = torch.tensor(shifted_intervention_locations)
        losing_intervention_locations = torch.tensor(shifted_intervention_locations)
    else:
        raise NotImplementedError(f"Positions {positions} not implemented")
    
    return {
        "winning_input_ids": winning_input_ids,
        "losing_input_ids": losing_input_ids,
        "winning_labels": winning_output_ids,
        "losing_labels": losing_output_ids,
        "winning_intervention_locations": winning_intervention_locations,
        "losing_intervention_locations": losing_intervention_locations,
        "prompt_lengths": torch.tensor(prompt_length - 1),
    }
    

def make_preference_data_module(
    tokenizer: transformers.PreTrainedTokenizer, dataset, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    preference_pairs=["orig_add"],
    prefix_tuning=False,
    steering_prompt_type="prepend",
    **kwargs
):
    """
    4-way preference training setup:

    - original instruction + steering:
        - winning: LLM Steered Response
        - losing: LM Response to Original Instruction

    - original instruction - steering:
        - winning: LM Response to Original Instruction
        - losing: LLM Steered Response

    - steered instruction + steering:
        - winning: LLM Steered Response
        - losing: LM Response to Original Instruction

    - steered instruction - steering:
        - winning: LM Response to Original Instruction
        - losing: LM Response to Steered Instruction
    """
    if not exclude_bos:
        prefix_length = 0

    all_data = {}
    for pair in preference_pairs:
        all_data[f"{pair}_winning_input_ids"] = []
        all_data[f"{pair}_losing_input_ids"] = []
        all_data[f"{pair}_winning_intervention_locations"] = []
        all_data[f"{pair}_losing_intervention_locations"] = []
        all_data[f"{pair}_winning_labels"] = []
        all_data[f"{pair}_losing_labels"] = []
        all_data[f"{pair}_prompt_lengths"] = []

    for item in dataset:        
        if f"{steering_prompt_type}_steered_input" not in item:
            input, winning_output, losing_output = item["question"], item["matching"], item["not_matching"]
            steered_input = None
            steered_output = None
        else:
            input, steered_input, winning_output, losing_output, steered_output = \
                item["question"], item[f"{steering_prompt_type}_steered_input"], item["matching"], item["not_matching"], item[f"{steering_prompt_type}_steered_output"]

        for pair in preference_pairs:
            if pair == "orig_add":
                new_data = preprocess_preference_data(
                    tokenizer, input, winning_output, losing_output, positions, prefix_length, prefix_tuning
                )
            elif pair == "orig_sub":
                new_data = preprocess_preference_data(
                    tokenizer, input, losing_output, winning_output, positions, prefix_length, prefix_tuning
                )
            elif pair == "steered_add":
                new_data = preprocess_preference_data(
                    tokenizer, steered_input, winning_output, losing_output, positions, prefix_length, prefix_tuning
                )
            elif pair == "steered_sub":
                new_data = preprocess_preference_data(
                    tokenizer, steered_input, losing_output, steered_output, positions, prefix_length, prefix_tuning
                )
            else:
                raise NotImplementedError(f"Preference pair {pair} not implemented")
            for k, v in new_data.items():
                all_data[f"{pair}_{k}"].append(v)
    
    train_dataset = datasets.Dataset.from_dict(all_data)
    train_dataset.set_format(
        type='torch', columns=list(all_data.keys()))
    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = PreferenceInterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn, preference_pairs=preference_pairs)
    
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


