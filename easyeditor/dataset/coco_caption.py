"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from .processor.llavaov_processors import LLaVAOneVisionProcessor
from .processor.qwen2vl_processors import Qwen2VLProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoProcessor

class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_name == "blip2" or config.model_name == "minigpt4":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
            if (config is not None and hasattr(config, 'tokenizer_name')):
                tok_name = (
                    config.tokenizer_name
                    if config.tokenizer_name is not None
                    else config.name
                )
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name, trust_remote_code=True
                )            
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token  
        elif "llava-onevision" in config.model_name.lower():  
            vis_processor = LLaVAOneVisionProcessor()
            tokenizer = AutoProcessor.from_pretrained(config.name)
        elif "qwen2-vl" in config.model_name.lower():
            vis_processor = Qwen2VLProcessor()
            tokenizer = AutoProcessor.from_pretrained(config.name)
        elif (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
        else:
            raise ValueError(f"Unknown model configuration: {config.model_name}")
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer:"

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]  
        for i, record in enumerate(self.annotation):
            if record['alt'] == "":
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            
            # image = Image.open(image_path).convert("RGB")
            # rephrase_image = Image.open(rephrase_image_path).convert("RGB")
            # locality_image = Image.open(locality_image_path).convert("RGB")

            image = self.vis_processor(image_path, file_type="image")
            rephrase_image = self.vis_processor(rephrase_image_path, file_type="image")  
            locality_image = self.vis_processor(locality_image_path, file_type="image")  
                      
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            item['file_type'] = "image"
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [" " + b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [" " + b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [" " + b['multimodal_locality_ground_truth'] for b in batch]
        
        model_name = self.config.model_name.lower()
        is_hf_multimodal = ("llava-onevision" in model_name) or ("qwen2-vl" in model_name)

        def _build_hf_multimodal_batch(prompts, targets, images, file_type):
            # prompts/targets are list[str], images is list of vision inputs or None
            if images is None:
                images_list = [None for _ in range(len(prompts))]
            else:
                images_list = images

            text_inputs = []
            for p, t, img in zip(prompts, targets, images_list):
                if file_type == "text":
                    chat = self.tok.apply_chat_template(
                        [{"role": "user", "content": [{"type": "text", "text": p}]}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    text_inputs.append(chat + t)
                elif file_type in ["image", "single-image", "multi-image"]:
                    num_images = len(img) if isinstance(img, list) else 1
                    content = [{"type": "image"}] * num_images + [{"type": "text", "text": p}]
                    chat = self.tok.apply_chat_template(
                        [{"role": "user", "content": content}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    if "qwen2-vl" in model_name and "|vision_start|" not in chat:
                        chat = "<|vision_start|><|image_pad|><|vision_end|>" + chat
                    text_inputs.append(chat + t)
                else:
                    raise AssertionError(f"Not support file type: {file_type}")

            processor_kwargs = {"text": text_inputs, "return_tensors": "pt", "padding": True}
            if file_type in ["image", "single-image", "multi-image"]:
                processor_kwargs["images"] = images_list
            multimodal_inputs = self.tok(**processor_kwargs)

            target_tokens = self.tok.tokenizer(
                targets,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                # max_length=multimodal_inputs["input_ids"].size(1),
            )["input_ids"]

            labels = torch.full_like(multimodal_inputs["input_ids"], -100)
            labels[:, -target_tokens.size(1) :] = target_tokens

            device = getattr(self.config, "device", None)
            target_dtype = getattr(self.config, "dtype", None)
            for k, v in multimodal_inputs.items():
                if isinstance(v, torch.Tensor):
                    if device is not None and torch.is_floating_point(v) and target_dtype is not None:
                        multimodal_inputs[k] = v.to(device=device, dtype=target_dtype)
                    elif device is not None:
                        multimodal_inputs[k] = v.to(device=device)

            multimodal_inputs["labels"] = labels.to(device=device) if device is not None else labels
            return multimodal_inputs

        if is_hf_multimodal:
            edit_inner = _build_hf_multimodal_batch(src, trg, image, "image")
            edit_outer = _build_hf_multimodal_batch(rephrase, trg, image, "image")
            edit_outer_image = _build_hf_multimodal_batch(src, trg, image_rephrase, "image")
            loc = _build_hf_multimodal_batch(loc_q, loc_a, None, "text")
            loc_image = _build_hf_multimodal_batch(
                [self.prompt.format(q) for q in m_loc_q], m_loc_a, m_loc_image, "image"
            )
        else:
            # edit_inner
            edit_inner = {}
            edit_inner['image'] = torch.stack(image, dim=0)
            edit_inner['text_input'] = [s + t for s, t in zip(src, trg)]
            edit_inner['labels'] = trg
            if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
                edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
                edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            else:
                edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
                edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
            # edit_outer
            edit_outer = {}
            edit_outer['image'] = torch.stack(image, dim=0)
            edit_outer['text_input'] = [r + t for r, t in zip(rephrase, trg)]
            edit_outer['labels'] = trg
            if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
                edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
                edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            else:
                edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
                edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
                
            # edit_outer_image
            edit_outer_image = {}
            edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
            edit_outer_image['text_input'] = [s + t for s, t in zip(src, trg)]
            edit_outer_image['labels'] = trg
            if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
                edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
                edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            else:
                edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
                edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
            # loc
            loc = {}
            loc['image'] = None
            loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
            loc['labels'] = loc_a
            if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
                loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
                loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            else:
                loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
                loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
            
            # m_loc
            loc_image = {}
            loc_image['image'] = torch.stack(m_loc_image, dim=0)
            loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
            loc_image['labels'] = m_loc_a
            if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
                loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
                loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            else:
                loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
                loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]


        # cond
        if is_hf_multimodal:
            # 对于 qwen2-vl 和 llava-onevision，使用 tokenizer 处理纯文本
            cond = self.tok.tokenizer(
                cond,
                return_tensors="pt",
                padding=True,
                # max_length=self.max_length,
                truncation=True,
            )
            device = getattr(self.config, "device", None)
            if device is not None:
                for k, v in cond.items():
                    if isinstance(v, torch.Tensor):
                        cond[k] = v.to(device=device)
        else:
            cond = self.tok(
                cond,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
