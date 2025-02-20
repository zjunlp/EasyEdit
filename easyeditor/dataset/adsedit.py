"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.llava_ov_processors import LLaVAOneVisionProcessor
from .processor.qwen2vl_processors import Qwen2VLProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoProcessor

class ADSDataset(BaseDataset):
    def __init__(self, 
                data_dir: str,
                size:  typing.Optional[int] = None, 
                config=None, 
                *args, 
                **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        model_name = config.model_name
        # get tokenizer and vis_processor
        if "llava" in model_name.lower():          
            self.vis_processor = LLaVAOneVisionProcessor()
        elif "qwen" in model_name.lower():
            self.vis_processor = Qwen2VLProcessor()
        else:
            raise AssertionError("Not support the vision processor of {}".format(model_name))
        
        self.tok = AutoProcessor.from_pretrained(model_name)          
        if self.tok.tokenizer.pad_token == None or self.tok.tokenizer.pad_token == '':
            self.tok.tokenizer.pad_token = self.tok.tokenizer.eos_token  
        self.config = config
        data = []
        f = open(data_dir, "r")
        self.annotation = json.load(f)
        if size is not None:
            self.annotation = self.annotation[:size]  
        for _, record in enumerate(self.annotation):
            
            if record['alt'] == "":
                continue
               
            image = record["image"]
            rephrase_image = record["image_rephrase"]
            locality_image = record['m_loc']
            file_type = record["image_type"]
            image = self.vis_processor(image, file_type=file_type)
            rephrase_image = self.vis_processor(rephrase_image, file_type=file_type)  
            locality_image = self.vis_processor(locality_image, file_type="image")  
                      
            item = {
                'prompt': record['src'],
                'type': file_type,
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image
            }

            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            data.append(item)
            
        if size is not None:
            data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
    
    def multimodal_tokenize(self, file_type, p, l, num_images=None):
        if file_type == "video":
            temp_prompt = self.tok.apply_chat_template([
                                    {

                                        "role": "user",
                                        "content": [
                                            {"type": "video"},
                                            {"type": "text", "text": p},
                                            ],
                                    },
                                ],
                                                add_generation_prompt=True,
                                                tokenize=False) + l
                        
            prompt_ids = self.tok.tokenizer(
            text=self.tok.apply_chat_template([
                            {

                                "role": "user",
                                "content": [
                                    {"type": "video"},
                                    {"type": "text", "text": p},
                                    ],
                            },
                        ],
                                    add_generation_prompt=True,
                                    tokenize=False), return_tensors="pt", padding=True, truncation=True)["input_ids"]

            
        elif file_type in ["image", "single-image", "multi-image"]:
            if file_type == "multi-image":
                num_images = num_images
            else:
                num_images = 1

            temp_prompt = self.tok.apply_chat_template([
                                    {

                                        "role": "user",
                                        "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}],
                                    },
                                ],
                                                add_generation_prompt=True,
                                                tokenize=False)  + l
            prompt_ids = self.tok.tokenizer(
            self.tok.apply_chat_template([
                            {

                                "role": "user",
                                "content": [
                                    [{"type": "image"}] * num_images + [{"type": "text", "text": p}]
                                    ],
                            },
                        ],
                                    add_generation_prompt=True,
                                    tokenize=False), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        elif file_type == "text":
            temp_prompt = self.tok.apply_chat_template([
                                    {

                                        "role": "user",
                                        "content": [{"type": "text", "text": p}],
                                    },
                                ],
                                                add_generation_prompt=True,
                                                tokenize=False)  + l
                            
            prompt_ids = self.tok.tokenizer(
            self.tok.apply_chat_template([
                            {

                                "role": "user",
                                "content": [{"type": "text", "text": p}],
                            },
                        ],
                                    add_generation_prompt=True,
                                    tokenize=False), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
            
        return temp_prompt, prompt_ids

    def collate_fn(self, batch):
        
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        
        # edit_inner
        edit_inner = {}
        edit_inner['type'] = batch[0]["type"]
        edit_inner['image'] = image
        edit_inner['text_input'] = []
        edit_inner['prompts_len'] = []
        for b in batch:
            temp_prompt, prompt_ids = self.multimodal_tokenize(b["type"], b["prompt"], b["target"], len(b["image"]))
            edit_inner['text_input'].append(temp_prompt)
            edit_inner['prompts_len'].append(len(prompt_ids[0]))
        
        edit_inner['labels'] = self.tok.tokenizer(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # edit_outer
        edit_outer = {}
        edit_outer['type'] = batch[0]["type"]
        edit_outer['image'] = image
        edit_outer['text_input'] = []
        edit_outer['prompts_len'] = []
        for b in batch:
            temp_prompt, prompt_ids = self.multimodal_tokenize(b["type"], b["rephrase_prompt"], b["target"], len(b["image"]))
            edit_outer['text_input'].append(temp_prompt)
            edit_outer['prompts_len'].append(len(prompt_ids[0]))
            
        edit_outer['labels'] = self.tok.tokenizer(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['type'] = batch[0]["type"]
        edit_outer_image['image'] = image_rephrase
        edit_outer_image['text_input'] = []
        edit_outer_image['prompts_len'] = []
        for b in batch:
            temp_prompt, prompt_ids = self.multimodal_tokenize(b["type"], b["prompt"], b["target"], len(b["image_rephrase"]))
            edit_outer_image['text_input'].append(temp_prompt)
            edit_outer_image['prompts_len'].append(len(prompt_ids[0]))
        
        edit_outer_image['labels'] = self.tok.tokenizer(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # loc
        loc = {}
        loc['type'] = "text"
        loc['image'] = None
        loc['text_input'] = []
        loc['prompts_len'] = []
        for b in batch:
            temp_prompt, prompt_ids = self.multimodal_tokenize("text", b["locality_prompt"], b["locality_ground_truth"])
            loc['text_input'].append(temp_prompt)
            loc['prompts_len'].append(len(prompt_ids[0]))
        
        loc['labels'] = self.tok.tokenizer(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['type'] = "image"
        loc_image['image'] = m_loc_image
        loc_image['text_input'] = []
        loc_image['prompts_len'] = []
        for b in batch:
            temp_prompt, prompt_ids = self.multimodal_tokenize("image", b["multimodal_locality_prompt"], b["multimodal_locality_ground_truth"])
            loc_image['text_input'].append(temp_prompt)
            loc_image['prompts_len'].append(len(prompt_ids[0]))
        loc_image['labels'] = self.tok.tokenizer(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image
        }
        return dict_to(batch, self.config.device)
