import os
import os.path as path
import sys
import json
import random
sys.path.append('..')
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
from sentence_transformers import SentenceTransformer
from easyeditor.dataset.longform import LongFormDataset

import argparse
import numpy as np

def eval(result_path):
    if path.exists(result_path):
        
        with open(result_path,'r') as file:
            datas=json.load(file)
        #data_rome_counterfact['post'].keys()  dict_keys(['rewrite_acc', 'locality', 'portability'])
        Edit_Succ_list=[data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
        print('Edit_Succ:',Edit_Succ)
        
        Portability_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['portability'].keys():
                case_list.append(sum(data_rome_counterfact['post']['portability'][key])/len(data_rome_counterfact['post']['portability'][key])*100)
            if len(case_list) != 0:
                Portability_list.append(np.mean(case_list))
        Overall_portability = np.mean(Portability_list)
        print('Overall_portability:',Overall_portability)

        Locality_list=[]
        for data_rome_counterfact in datas:
            case_list=[]
            for key in data_rome_counterfact['post']['locality'].keys():
                case_list.append(sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:',Overall_locality)
        
        Fluency_list=[x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency=sum(Fluency_list)/len(Fluency_list)*100
        print('Fluency:',Fluency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default='longform', type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./lf_pre.json', type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    else:
        raise NotImplementedError
    
    # 加载数据集
    datas = LongFormDataset(args.data_dir, size=args.ds_size)
    
    # 提取基本编辑字段
    basic_prompts = [data['prompt_full'] for data in datas]
    basic_targets = [data['target_new'] for data in datas]
    subjects = [data['subject'] for data in datas]
    
    # 处理可迁移性测试数据
    portability_paraphrase_prompts = []
    portability_paraphrase_ans = []
    
    portability_generation_prompts = []
    portability_generation_ans = []

    for data in datas:
        # 处理 paraphrase_prompts
        if "paraphrase_prompts" in data and len(data['paraphrase_prompts']) > 0:
            temp_prompts = data['paraphrase_prompts']
            temp_answers = [data['target_new']] * len(temp_prompts)
            portability_paraphrase_prompts.append(temp_prompts)
            portability_paraphrase_ans.append(temp_answers)
        else:
            portability_paraphrase_prompts.append([])
            portability_paraphrase_ans.append([])
        
        # 处理 generation_prompts
        if "generation_prompts" in data and len(data['generation_prompts']) > 0:
            temp_prompts = data['generation_prompts']
            temp_answers = [data['target_new']] * len(temp_prompts)
            portability_generation_prompts.append(temp_prompts)
            portability_generation_ans.append(temp_answers)
        else:
            portability_generation_prompts.append([])
            portability_generation_ans.append([])
    
    # 处理局部性测试数据
    locality_neighborhood_prompts = []
    locality_neighborhood_ans = []
    
    for data in datas:
        # 处理 neighborhood_prompts
        if "neighborhood_prompts" in data and len(data['neighborhood_prompts']) > 0:
            temp_prompts = data['neighborhood_prompts']
            temp_answers = [data['target_true']] * len(temp_prompts)
            locality_neighborhood_prompts.append(temp_prompts)
            locality_neighborhood_ans.append(temp_answers)
        else:
            locality_neighborhood_prompts.append([])
            locality_neighborhood_ans.append([])
    
    # 构建评测输入结构
    portability_inputs = {
        'Paraphrase_Prompts': {
            'prompt': portability_paraphrase_prompts,
            'ground_truth': portability_paraphrase_ans
        },
        'Generation_Prompts': {
            'prompt': portability_generation_prompts,
            'ground_truth': portability_generation_ans
        }
    }
    
    locality_inputs = {
        'Neighborhood_Prompts': {
            'prompt': locality_neighborhood_prompts,
            'ground_truth': locality_neighborhood_ans
        }
    }
    
    # 打印数据处理结果统计
    print(f"\n===== 数据处理结果 =====")
    print(f"基本提示数量: {len(basic_prompts)}")
    print(f"可移植性测试 Paraphrase_Prompts: {sum(1 for prompts in portability_paraphrase_prompts if len(prompts) > 0)}/{len(portability_paraphrase_prompts)}")
    print(f"可移植性测试 Generation_Prompts: {sum(1 for prompts in portability_generation_prompts if len(prompts) > 0)}/{len(portability_generation_prompts)}")
    print(f"局部性测试 Neighborhood_Prompts: {sum(1 for prompts in locality_neighborhood_prompts if len(prompts) > 0)}/{len(locality_neighborhood_prompts)}")
    
    # 加载超参数
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    args.pre_file = f"./{hparams.model_name.split('/')[-1]}_{args.datatype}_pre_edit.json"
    print(args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        assert len(pre_edit) == len(basic_prompts)
    else:
        pre_edit = None
        
    # 处理特定编辑方法
    if args.editing_method == 'IKE':
        train_ds = LongFormDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    elif args.editing_method == 'ICE':
        hparams.use_icl_examples = False
        train_ds = None
    else:
        train_ds = None
    
    # 初始化编辑器
    editor = BaseEditor.from_hparams(hparams)
    
    # 执行编辑并评测
    metrics, final_model, _ = editor.edit(
        prompts=basic_prompts,
        target_new=basic_targets,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        pre_file=args.pre_file,
        pre_edit=pre_edit,
        test_generation=True,
        train_ds=train_ds
    )

    # 保存和评估结果
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_results.json')
    json.dump(metrics, open(result_path, 'w'), indent=4)
    eval(result_path)