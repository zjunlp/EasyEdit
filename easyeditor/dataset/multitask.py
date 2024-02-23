import json
import random
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer
from ..util.globals import *
from ..trainer.utils import dict_to

def add_gpt_sep(tokenizer, model):
    tokenizer.add_special_tokens({'sep_token': '</s>'})
    model.resize_token_embeddings(len(tokenizer))
    model.lm_head.weight.data[-1, :] = model.lm_head.weight.data.mean(0)
    
class MultiTaskDataset(Dataset):

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        multi_task = data_dir

        if(config is not None):
            self.config = config
        if(config is not None and hasattr(config, 'max_length')):
            self.max_length = config.max_length
        else:
            self.max_length = 128

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
                #   "Steer the chatbot towards a [LABEL] attitude about [TOPIC], but make sure it doesn't sway its stance elsewhere.", ## The last one for testing instruction generality.
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
                #   "A dataset curated to push the boundaries of model's capabilities in recognizing and processing tail entities.",
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
                #   "A dataset focusing on the integration of newly verified factual data from WikiData.",
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
                #   "A dataset for driving the development of AI that can predict objects given a subject and its relation.",
              ]
        }
        
        # For Meta Training
        if(config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name
            )
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                tokenizer.add_special_tokens({'sep_token': '</s>'})
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        with open(multi_task, "r") as f:
            raw = json.load(f)

        from random import choice
        random.seed(0)

        data = []
        for i, record in enumerate(raw):
            if record["target_new"] == "":
                continue
            assert 'type' in record.keys(), 'type not found in data'
            task = record['type']
            if task == 'convsent':
                description = choice(desc[task]).replace("[LABEL]", "positive" if "positively" in record["target_new"] else "negative").replace("[TOPIC]", record['subject'])
                template = temp.format(task, description, record['prompt'])
            else:
                description = choice(desc[task])
                template = temp.format(task, description, record['prompt'])
                
            request = {
                    "case_id": i,
                    "subject": record["subject"],
                    "prompt": template,
                    "target_new": record["target_new"],
                    "metric_kwargs": record["metric_kwargs"] if "metric_kwargs" in record.keys() else None,
                }   
            if "locality" in record.keys() and record["locality"]:
                request["locality"] = {}
                request["locality"]["prompt"] = []
                request["locality"]["ground_truth"] = []
                for locality_key in record["locality"].keys():
                    prompt = []
                    ground_truth = []
                    if isinstance(record["locality"][locality_key], list):
                        for item in record["locality"][locality_key]:
                            prompt += [item["prompt"]]
                            ground_truth += [choice(choice(item["ground_truth"]))]
                        request["locality"]["prompt"] += prompt
                        request["locality"]["ground_truth"] += ground_truth
                    else:
                        request["locality"]["prompt"] += record["locality"][locality_key]["prompt"]
                        request["locality"]["ground_truth"] += record["locality"][locality_key]["ground_truth"]
                        
            if "portability" in record.keys() and record["portability"]:
                request["portability"] = {}
                request["portability"]["prompt"] = []
                request["portability"]["ground_truth"] = []
                for portability_key in record["portability"].keys():
                    prompt = []
                    ground_truth = []
                    if isinstance(record["portability"][portability_key], list):
                        for item in record["portability"][portability_key]:
                            prompt += [item["prompt"]]
                            ground_truth += [choice(choice(item["ground_truth"]))]
                        request["portability"]["prompt"] += prompt
                        request["portability"]["ground_truth"] += ground_truth
                    else:
                        request["portability"]["prompt"] += record["portability"][portability_key]["prompt"]
                        request["portability"]["ground_truth"] += record["portability"][portability_key]["ground_truth"]
            
            data.append(request)

        if size is not None:
            data = data[:size]
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [' ' + b["target_new"] for b in batch] # alter

        src = [src_ + trg_ for src_, trg_ in zip(src, trg)]
        
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                # "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        # locality
        is_loc = False
        if "locality" in batch[0].keys():
            is_loc = True
            loc = []
            loc_ans = []
            for b in batch:
                loc += b["locality"]["prompt"]
                loc_ans += [' ' + i for i in b["locality"]["ground_truth"]]
            loc = [loc_ + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]
            loc = dict(
                self.tok(
                    loc,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )

            loc_ans = dict(
                self.tok(
                    loc_ans,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )
            loc["decoder_attention_mask"] = loc_ans["attention_mask"]
            loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])   
        elif batch[0]["metric_kwargs"]:
            is_loc = True
            metric_kwargs = batch[0]["metric_kwargs"]
            same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])])
            batch[0]["metric_kwargs"]["same_mask"] = same_mask
            edit_toks = {
                f"{k1}_{k2}": v2
                for k1, v1 in {
                    "inner": metric_kwargs["inner_all_qa"],
                    "outer": metric_kwargs["outer_all_qa"],
                }.items()
                for k2, v2 in self.tok(
                    v1,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                ).items()
            }
            for key in ["inner", "outer"]:
                value = edit_toks[f"{key}_input_ids"]
                mask = [([True] * value.shape[-1])] * value.shape[0]
                for i in range(value.shape[0]):
                    sep_idx = list(value[i]).index(self.tok.convert_tokens_to_ids("</s>"))
                    for j in range(sep_idx): # mask </s>
                        mask[i][j] = False
                edit_toks[key + "_q_mask"] = torch.tensor(mask)
                edit_toks[key + "_labels"] = self.get_edit_labels(edit_toks[key + "_input_ids"])
                if key == "outer":
                    loc = {
                        "input_ids": edit_toks["outer_input_ids"],
                        "attention_mask": edit_toks["outer_attention_mask"],
                        "labels": edit_toks["outer_labels"],
                        "q_mask": edit_toks["outer_q_mask"],
                    }
                elif key == "inner":
                    edit_inner = {
                        "input_ids": edit_toks["inner_input_ids"],
                        "attention_mask": edit_toks["inner_attention_mask"],
                        "labels": edit_toks["inner_labels"],
                    }
        
        # portability
        is_port = False      
        if "portability" in batch[0].keys():
            is_port = True
            port = []
            port_ans = []
            for b in batch:
                port += b["portability"]["prompt"]
                port_ans += [' ' + i for i in b["portability"]["ground_truth"]]
            port = [port_ + port_ans_ for port_, port_ans_ in zip(port, port_ans)]
            port = dict(
                self.tok(
                    port,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )

            port_ans = dict(
                self.tok(
                    port_ans,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            )
            port["decoder_attention_mask"] = port_ans["attention_mask"]
            port["labels"] = self.get_edit_labels(port_ans["input_ids"])

        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": None,
            "loc": loc if is_loc else None,
            "port": port if is_port else None,
            "raw": batch,
            "metric_kwargs": metric_kwargs if batch[0]["metric_kwargs"] else None,
        }
        return dict_to(batch, self.config.device)
