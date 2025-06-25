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
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
import json
import numpy as np
import pdb

# more examples and application are in https://github.com/yaohui120/ComprehendEdit
class ComprehendEditDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
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
        if config.model_name.lower() == 'llava1.5':
            from transformers import CLIPProcessor, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False)
            vis_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root)
        
        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.mode = 'train'
        if 'mode' in kwargs.keys():
            self.mode = kwargs['mode']
        save_dir = './data_our'

        self.prompt = {'gqa': "Question: {} Short answer:",
                       'tallyqa': "Question: {} Answer with a number. Short answer:",
                       'vsr': "Question: Is this description true or false? Description: {} Short answer:",
                       'textvqa': "Question: {} Short answer:",
                       'mathvista': "{} Short answer:"}

        self.annotation = [json.loads(q) for q in open(os.path.join(data_dir,'ComprehendEdit_{}.json'.format(self.mode)), "r")][0]

        self.topk = -1 # no topk in kwargs.keys() means we didn't measure KPI and KGI
        if 'topk' in kwargs.keys():
            self.topk = kwargs['topk']

        data = []
        if size is not None: # size is None when testing
            self.annotation = self.annotation[:size]
        
        ind = []
        if self.mode == 'train' and ('diverse' in kwargs.keys() and kwargs['diverse']):
            ind = self.get_diverse_data() # only for training set
            path = os.path.join(save_dir, 'ComprehendEdit_{}_{}_diverse.pth'.format(self.mode, config.model_name.lower()))
        else:
            path = os.path.join(save_dir, 'ComprehendEdit_{}_{}.pth'.format(self.mode, config.model_name.lower()))
        if self.mode != 'train' and config.sequence_len > 0:
            path = os.path.join(save_dir, 'ComprehendEdit_{}_{}_continual.pth'.format(self.mode, config.model_name.lower()))
        self.image = True
        
        if os.path.exists(path):
            data = torch.load(path)
            print('Loaded datasets from {}'.format(path))
        else:
            for i, record in enumerate(self.annotation):
                if self.mode == 'train' and len(ind)!=0 and i not in ind:
                    continue
                if record['answer'] == "":
                    continue
                if i > 0 and i % 1000 == 0:
                    print('Loaded {} samples'.format(i))
                
                image_path = os.path.join(data_dir, record["image"])
                # rephrase_image_path = image_path
                locality_image_path = os.path.join(data_dir, record["multimodal_locality_image"])
                
                image = image_path
                # rephrase_image = image
                locality_image = locality_image_path
                if self.image:
                    image = Image.open(image_path).convert("RGB")
                    # rephrase_image = Image.open(rephrase_image_path).convert("RGB")
                    locality_image = Image.open(locality_image_path).convert("RGB")

                    image = self.vis_processor(image)
                    # rephrase_image = image
                    locality_image = self.vis_processor(locality_image)
                    
                    
                item = {
                    'pid': record['pid'],
                    'prompt': record['question'],
                    'pred': '',
                    'target': record['answer'],
                    'rephrase_prompt': record['rephrase'],
                    'image': image,
                    # 'image_rephrase': rephrase_image,
                    'locality_prompt': record['locality_prompt'],
                    'locality_ground_truth': record['locality_prompt'],
                    'multimodal_locality_image': locality_image,
                    'multimodal_locality_prompt': record['multimodal_locality_prompt'],
                    'multimodal_locality_ground_truth': record['multimodal_locality_ground_truth'],
                    'cond': "{} >> {} || {}".format(
                        '',
                        record['answer'],
                        record['question']
                    ),
                    'image_path': image_path,
                    'task': record['Category'],
                    'source': record['source']
                }

                if self.mode != 'train':
                    item['img_topk'] = record['img_topk'][:self.topk]
                    item['txt_topk'] = record['txt_topk'][:self.topk]
                    item['img_last_topk'] = record['img_last_topk'][-self.topk:]
                    item['txt_last_topk'] = record['txt_last_topk'][-self.topk:]
                    
                    item['ori_rt_img_topk'] = record['ori_rt_img_topk'][:self.topk]
                    item['ori_rt_txt_topk'] = record['ori_rt_txt_topk'][:self.topk]
                    item['ori_rt_img_last_topk'] = record['ori_rt_img_last_topk'][-self.topk:]
                    item['ori_rt_txt_last_topk'] = record['ori_rt_txt_last_topk'][-self.topk:]
                    
                data.append(item)
            torch.save(data, path)

        if not hasattr(self.config, 'alg'):
            self.config.alg = self.config.alg_name
        if self.mode != 'train' and self.topk != -1 and size is None and self.config.alg.lower() in ['ike', 'ft', 'serac_multi', 'mend', 'hice', 'melo', 'elder', 'recem']:
            self.test_ = True
        else:
            self.test_ = False

        if self.test_:
            path = os.path.join(save_dir, 'ComprehendEdit_inner_{}.pth'.format(config.model_name.lower()))
            if os.path.exists(path):
                self.all_edit_inner = torch.load(path)
            else:
                self.all_edit_inner = []
                test_data = json.load(open(os.path.join(data_dir,'ComprehendEdit_test.json'), "r"))
                for step, sample in enumerate(data):
                    image_path = sample['image_path']
                    image = Image.open(image_path).convert("RGB")
                    image = self.vis_processor(image)
                    
                    src, trg, image = [sample['prompt']], [sample['target']], [image]
                    sources = [sample['source'].lower()]
                    
                    edit_inner = {}
                    edit_inner['image'] = torch.stack(image, dim=0)
                    edit_inner['text_input'] = [self.prompt[source].format(s) + t for s, t, source in zip(src, trg, sources)]
                    edit_inner['text_labels'] = trg
                    edit_inner['image_path'] = image_path
                    edit_inner['prompt'] = src
                    edit_inner['target'] = trg
                    edit_inner['ori_pred_blip2'] = np.array(test_data[step]['ori_pred_blip2'])
                    edit_inner['ori_pred_minigpt4'] = np.array(test_data[step]['ori_pred_minigpt4'])
                    edit_inner['prompts_len'], edit_inner['labels'] = self.get_prompt_len(src, trg, sources)
                    self.all_edit_inner.append(edit_inner)
                torch.save(self.all_edit_inner, path)

            path = os.path.join(save_dir, 'ComprehendEdit_ori_right_{}.pth'.format(config.model_name.lower()))
            if os.path.exists(path):
                self.ori_right = torch.load(path)
            else:
                self.ori_right = []
                right_data = json.load(open(os.path.join(data_dir,'ComprehendEdit_ori_right.json'), "r"))
                for sample in right_data:
                    image_path = os.path.join(data_dir, sample["image"])
                    image = Image.open(image_path).convert("RGB")
                    image = self.vis_processor(image)
                    
                    src, trg, image = [sample['question']], [sample['answer']], [image]
                    sources = [sample['source'].lower()]
                    
                    edit_inner = {}
                    edit_inner['image'] = torch.stack(image, dim=0)
                    edit_inner['text_input'] = [self.prompt[source].format(s) + t for s, t, source in zip(src, trg, sources)]
                    edit_inner['labels'] = trg
                    edit_inner['text_labels'] = trg
                    edit_inner['image_path'] = image_path
                    edit_inner['prompt'] = src
                    edit_inner['target'] = trg
                    edit_inner['ori_pred_blip2'] = np.array(sample['ori_pred_blip2'])
                    edit_inner['ori_pred_minigpt4'] = np.array(sample['ori_pred_minigpt4'])
                    edit_inner['prompts_len'], edit_inner['labels'] = self.get_prompt_len(src, trg, sources)
                    self.ori_right.append(edit_inner)
                torch.save(self.ori_right, path)
        self._data = data
        del self.annotation

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def get_diverse_data(self, topk = 50):
        import numpy as np
        from scipy.spatial.distance import cdist
        
        results = torch.load(f'./{self.config.task.lower()}_train_img_txt_fea.pth')
        lambda_ = 0.5
        img_fea = (results['img_feas'].T / torch.norm(results['img_feas'].T, dim=-1).unsqueeze(-1)).T
        txt_fea = (results['txt_feas'].T / torch.norm(results['txt_feas'].T, dim=-1).unsqueeze(-1)).T
        # fea = lambda_*img_fea + (1-lambda_)*txt_fea
        fea = torch.cat((img_fea, txt_fea), dim=-1)
        fea = fea / torch.norm(fea, dim=-1).unsqueeze(-1)
        num_clusters = fea.shape[0]//20 # 5%
        
        path = f'kmeans_results_{self.config.task.lower()}_{num_clusters}_cat_fea.pth'
        if os.path.exists(path):
            km_cluster = torch.load(path)
            all_id = km_cluster.predict(fea)
        else:
            print('Begin clustering...')
            from sklearn.cluster import KMeans
            km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++')
            all_id = km_cluster.fit_predict(fea) # about 20-40 mins
            torch.save(km_cluster, path)
        
        centers = torch.tensor(km_cluster.cluster_centers_)
        centers = centers / torch.norm(centers, dim=-1).unsqueeze(-1)
        res = []
        for class_id in range(num_clusters):
            ind = np.where(all_id == class_id)[0]
            # dists = torch.norm(fea[ind]-centers[class_id], p=2, dim=-1)
            # ind1 = torch.sort(dists, descending=False)[1]
            np.random.seed(1993)
            ind1 = np.random.permutation(len(ind))
            res.append(ind[ind1[0]])
        
        return res

    def process_img(self, img):
        if self.config.model_name.lower() == 'llava1.5':
            image = self.vis_processor(images=img, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.vis_processor(img)
        return image
        
    def get_prompt_len(self, src, trg, sources, imgs=True):
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            prompts_len = [len(self.tok.encode(self.prompt[source].format(s), add_special_tokens=False)) for s, source in zip(src, sources)]
            labels = self.tok(trg, add_special_tokens=False, return_tensors="pt",)['input_ids']
        elif self.config.model_name == "llava1.5":
            prompts_len = [len(self.tok.encode(self.prompt[source].format(s), padding=True, return_tensors="pt",)[0]) for s, source in zip(src, sources)]
            labels = self.tok(trg, padding=True, return_tensors="pt",)['input_ids']
        else:
            prompts_len = [len(self.tok.encode(self.prompt[source].format(s))) for s, source in zip(src, sources)]
            labels = self.tok(trg, return_tensors="pt",)['input_ids']
        return prompts_len, labels

    def collate_fn(self, batch):
        self.config.device = 0 # should be consistant with yaml
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        # image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        sources = [b['source'].lower() for b in batch]
        
        # edit_inner
        edit_inner = {}
        if self.image:
            edit_inner['image'] = torch.stack(image, dim=0)
        else:
            edit_inner['image'] = image
        edit_inner['prompts_len_input'] = [self.prompt[source].format(s) for s, source in zip(src, sources)]
        edit_inner['prompts_len'], edit_inner['labels'] = self.get_prompt_len(src, trg, sources)
        edit_inner['image_path'] = [b['image_path'] for b in batch]
        edit_inner['ori_qs'] = src
        edit_inner['pid'] = [b['pid'] for b in batch]
        edit_inner['text_input'] = [self.prompt[source].format(s) + t for s, t, source in zip(src, trg, sources)]
        edit_inner['text_labels'] = trg
        edit_inner['prompt'] = src
        edit_inner['target'] = trg
        edit_inner['rephrase_prompt'] = rephrase
        edit_inner['rephrase_text_input'] = [self.prompt[source].format(r) + t for r, t, source in zip(rephrase, trg, sources)]
        edit_inner['locality_prompt'] = loc_q
        edit_inner['locality_ground_truth'] = loc_a
        edit_inner['source'] = sources
        edit_inner['cat'] = 'edit'
        edit_inner['target_new'] = trg
        
        if self.mode != 'train' and self.topk != -1 and self.test_:
            img_topk, txt_topk = [b['img_topk'][:self.topk] for b in batch], [b['txt_topk'][:self.topk] for b in batch]
            edit_inner['img_topk'], edit_inner['txt_topk'] = [], []
            img_last_topk, txt_last_topk = [b['img_last_topk'][-self.topk:] for b in batch], [b['txt_last_topk'][-self.topk:] for b in batch]
            edit_inner['img_last_topk'], edit_inner['txt_last_topk'] = [], []
            for ind_img, ind_txt, ind_last_img, ind_last_txt in zip(img_topk, txt_topk, img_last_topk, txt_last_topk):
                for i, j, k, m in zip(ind_img, ind_txt, ind_last_img, ind_last_txt):
                    for x in [i, j, k, m]:
                        self.all_edit_inner[x]['cat'] = 'edit'
                    edit_inner['img_topk'].append(self.all_edit_inner[i])
                    edit_inner['txt_topk'].append(self.all_edit_inner[j])
                    edit_inner['img_last_topk'].append(self.all_edit_inner[k])
                    edit_inner['txt_last_topk'].append(self.all_edit_inner[m])
            
            img_topk, txt_topk = [b['ori_rt_img_topk'][:self.topk] for b in batch], [b['ori_rt_txt_topk'][:self.topk] for b in batch]
            edit_inner['ori_rt_img_topk'], edit_inner['ori_rt_txt_topk'] = [], []
            img_last_topk, txt_last_topk = [b['ori_rt_img_last_topk'][-self.topk:] for b in batch], [b['ori_rt_txt_last_topk'][-self.topk:] for b in batch]
            edit_inner['ori_rt_img_last_topk'], edit_inner['ori_rt_txt_last_topk'] = [], []
            for ind_img, ind_txt, ind_last_img, ind_last_txt in zip(img_topk, txt_topk, img_last_topk, txt_last_topk):
                for i, j, k, m in zip(ind_img, ind_txt, ind_last_img, ind_last_txt):
                    for x in [i, j, k, m]:
                        self.ori_right[x]['cat'] = 'edit'
                    edit_inner['ori_rt_img_topk'].append(self.ori_right[i])
                    edit_inner['ori_rt_txt_topk'].append(self.ori_right[j])
                    edit_inner['ori_rt_img_last_topk'].append(self.ori_right[k])
                    edit_inner['ori_rt_txt_last_topk'].append(self.ori_right[m])

        # edit_outer
        edit_outer = {}
        if self.image:
            edit_outer['image'] = torch.stack(image, dim=0)
        else:
            edit_outer['image'] = image
        edit_outer['prompts_len_input'] = [self.prompt[source].format(r) for r, source in zip(rephrase, sources)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'], edit_outer['labels'] = self.get_prompt_len(rephrase, trg, sources)
        edit_outer['cat'] = 'rephrase'
        edit_outer['text_input'] = [self.prompt[source].format(r) + t for r, t, source in zip(rephrase, trg, sources)]
        edit_outer['text_labels'] = trg
        edit_outer['prompt'] = rephrase
        edit_outer['target'] = trg
        edit_outer['source'] = sources
        
        # generated image content is not consist with question, so we don't use these samples.
        # # edit_outer_image
        # edit_outer_image = {}
        # edit_outer_image['image'] = torch.stack(image_rephrase, dim=0).to(self.config.device)
        # edit_outer_image['text_input'] = [self.prompt[source].format(s) + t for s, t, source in zip(src, trg, sources)]
        # edit_outer_image['labels'] = trg
        # if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
        #     edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt[source].format(s), add_special_tokens=False)) for s, source in zip(src, sources)]
        #     edit_outer_image['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        # else:
        #     edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt[source].format(s))) for s, source in zip(src, sources)]
        #     edit_outer_image['labels'] = self.tok.encode(trg, return_tensors="pt",)
        
        # loc
        loc = {}
        loc['image'], loc['image_path'] = None, None
        for tt in loc_q:
            if '?' != tt[-1]:
                tt += '?'
        loc['image'] = None
        loc['prompts_len'], loc['labels'] = self.get_prompt_len(loc_q, loc_a, ['other' for q in loc_q], imgs=False)
        loc['locality_prompt'] = loc_q
        loc['prompts_len_input'] = [q for q in loc_q]
        # loc['text_input'] = [self.prompt['gqa'].format(q) + a for q, a in zip(loc_q, loc_a)]
        loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
        loc['prompt'] = loc_q
        loc['target'] = loc_a
        loc['cat'] = 'locality_prompt'
        
        # m_loc
        loc_image = {}
        if self.image:
            loc_image['image'] = torch.stack(m_loc_image, dim=0).to(self.config.device)
        else:
            loc_image['image'] = m_loc_image
        loc_image['prompts_len'], loc_image['labels'] = self.get_prompt_len(m_loc_q, m_loc_a, ['gqa' for q in m_loc_q])
        loc_image['multimodal_locality_prompt'] = m_loc_q
        loc_image['prompts_len_input'] = [self.prompt['gqa'].format(q) for q in m_loc_q]
        loc_image['text_input'] = [self.prompt['gqa'].format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['prompt'] = m_loc_q
        loc_image['target'] = m_loc_a
        loc_image['cat'] = 'multimodal_locality_image'

        # cond
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
            # "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }

        return dict_to(batch, self.config.device)
