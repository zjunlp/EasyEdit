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
from easyeditor.dataset.counterfact import CounterFactDataset

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
    parser.add_argument('--datatype', default='counterfact',type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./cf_pre.json', type=str)

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
    datas = CounterFactDataset(args.data_dir, size=args.ds_size)
    
    # 提取基本编辑和描述性编辑字段
    basic_prompts = [data['prompt'] for data in datas]
    basic_targets = [data['target_new'] for data in datas]
    descriptive_prompts = [data.get('descriptive_prompt', '') for data in datas]
    descriptive_targets = [data.get('descriptive_target', '') for data in datas]
    subjects = [data['subject'] for data in datas]
    
    # 处理可迁移性测试数据
    portability_Subject_Aliasing_prompts = []
    portability_Subject_Aliasing_ans = []

    for data in datas:
        # 统一处理可移植性数据 - 适用于MQuAKE-CF和WikiUpdate
        if "portability_data" in data and len(data['portability_data']) > 0:
            temp_prompts = []
            temp_answers = []
            portability_data = data.get("portability_data", [])
            portability_answer = data.get("portability_answer", data['target_new'])
            
            for question in portability_data:
                if question and question.strip() != "":
                    temp_prompts.append(question)
                    temp_answers.append(portability_answer)
            
            # 只有当有有效提示时才添加
            if temp_prompts:
                portability_Subject_Aliasing_prompts.append(temp_prompts)
                portability_Subject_Aliasing_ans.append(temp_answers)
            else:
                portability_Subject_Aliasing_prompts.append([])
                portability_Subject_Aliasing_ans.append([])
        # 优先使用KnowEdit格式的数据
        elif data.get('portability_s'):
            temp_prompts = []
            temp_answers = []
            for pr in data['portability_s']:
                if pr and isinstance(pr, dict) and 'prompt' in pr and 'ground_truth' in pr:
                    temp_prompts.append(pr['prompt'])
                    temp_answers.append(pr['ground_truth'])
            portability_Subject_Aliasing_prompts.append(temp_prompts)
            portability_Subject_Aliasing_ans.append(temp_answers)
        else:
            portability_Subject_Aliasing_prompts.append([])
            portability_Subject_Aliasing_ans.append([])
    
    # 处理局部性测试数据
    locality_Relation_Specificity_prompts = []
    locality_Relation_Specificity_ans = []
    
    for data in datas:
        # 优先使用KnowEdit格式的数据
        if data.get('locality_rs'):
            temp_prompts = []
            temp_answers = []
            for pr in data['locality_rs']:
                if pr and isinstance(pr, dict) and 'prompt' in pr and 'ground_truth' in pr:
                    temp_prompts.append(pr['prompt'])
                    temp_answers.append(pr['ground_truth'])
            locality_Relation_Specificity_prompts.append(temp_prompts)
            locality_Relation_Specificity_ans.append(temp_answers)
        # 使用统一格式处理locality_data (适用于MQuAKE-CF和WikiUpdate)
        elif data.get('locality_data') and len(data['locality_data']) > 0:
            temp_prompts = []
            temp_answers = []
            for item in data['locality_data']:
                if 'prompt' in item and 'subject' in item and 'target' in item:
                    try:
                        prompt = f"Complete the following: {item['prompt'].format(item['subject'])}"
                        target = item['target']
                        if target and target.strip() != "":
                            temp_prompts.append(prompt)
                            temp_answers.append(target)
                    except Exception as e:
                        print(f"警告：格式化提示时出错: {e}")
                        continue
            # 只有当有有效提示时才添加
            if temp_prompts:
                locality_Relation_Specificity_prompts.append(temp_prompts)
                locality_Relation_Specificity_ans.append(temp_answers)
            else:
                locality_Relation_Specificity_prompts.append([])
                locality_Relation_Specificity_ans.append([])
        else:
            locality_Relation_Specificity_prompts.append([])
            locality_Relation_Specificity_ans.append([])
    
    # 构建评测输入结构
    portability_inputs = {
        'Subject_Aliasing': {
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        }
    }
    
    locality_inputs = {
        'Relation_Specificity': {
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        }
    }
    
    # 添加更多可移植性和局部性测试（与KnowEdit格式兼容）
    # 添加可迁移性测试 - reasoning
    portability_reasoning_prompts = []
    portability_reasoning_ans = []
    
    for data in datas:
        if data.get('portability_r'):
            temp_prompts = []
            temp_answers = []
            for pr in data['portability_r']:
                if pr and isinstance(pr, dict) and 'prompt' in pr and 'ground_truth' in pr:
                    temp_prompts.append(pr['prompt'])
                    temp_answers.append(pr['ground_truth'])
            portability_reasoning_prompts.append(temp_prompts)
            portability_reasoning_ans.append(temp_answers)
        else:
            portability_reasoning_prompts.append([])
            portability_reasoning_ans.append([])
            
    # 添加可迁移性测试 - logical generalization
    portability_logical_prompts = []
    portability_logical_ans = []
    
    for data in datas:
        if data.get('portability_l'):
            temp_prompts = []
            temp_answers = []
            for pr in data['portability_l']:
                if pr and isinstance(pr, dict) and 'prompt' in pr and 'ground_truth' in pr:
                    temp_prompts.append(pr['prompt'])
                    temp_answers.append(pr['ground_truth'])
            portability_logical_prompts.append(temp_prompts)
            portability_logical_ans.append(temp_answers)
        else:
            portability_logical_prompts.append([])
            portability_logical_ans.append([])
            
    # 添加局部性测试 - forgetfulness
    locality_forgetfulness_prompts = []
    locality_forgetfulness_ans = []
    
    for data in datas:
        if data.get('locality_f'):
            temp_prompts = []
            temp_answers = []
            for pr in data['locality_f']:
                if pr and isinstance(pr, dict) and 'prompt' in pr and 'ground_truth' in pr:
                    temp_prompts.append(pr['prompt'])
                    temp_answers.append(pr['ground_truth'])
            locality_forgetfulness_prompts.append(temp_prompts)
            locality_forgetfulness_ans.append(temp_answers)
        else:
            locality_forgetfulness_prompts.append([])
            locality_forgetfulness_ans.append([])
    
    # 更新评测输入结构，添加更多测试类型
    if any(len(prompts) > 0 for prompts in portability_reasoning_prompts):
        portability_inputs['reasoning'] = {
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans
        }
        
    if any(len(prompts) > 0 for prompts in portability_logical_prompts):
        portability_inputs['Logical_Generalization'] = {
            'prompt': portability_logical_prompts,
            'ground_truth': portability_logical_ans
        }
        
    if any(len(prompts) > 0 for prompts in locality_forgetfulness_prompts):
        locality_inputs['Forgetfulness'] = {
            'prompt': locality_forgetfulness_prompts,
            'ground_truth': locality_forgetfulness_ans
        }
    
    # 打印数据处理结果统计
    print(f"\n===== 数据处理结果 =====")
    print(f"基本提示数量: {len(basic_prompts)}")
    print(f"可移植性测试 Subject_Aliasing: {sum(1 for prompts in portability_Subject_Aliasing_prompts if len(prompts) > 0)}/{len(portability_Subject_Aliasing_prompts)}")
    print(f"局部性测试 Relation_Specificity: {sum(1 for prompts in locality_Relation_Specificity_prompts if len(prompts) > 0)}/{len(locality_Relation_Specificity_prompts)}")
    
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
        train_ds = CounterFactDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    elif args.editing_method == 'ICE':
        hparams.use_icl_examples = False
        train_ds = None
    else:
        train_ds = None
    
    # 初始化编辑器
    editor = BaseEditor.from_hparams(hparams)
    
    # 检查是否有描述性编辑数据
    has_descriptive_data = all(desc and desc.strip() for desc in descriptive_prompts) and all(desc and desc.strip() for desc in descriptive_targets)
    
    if has_descriptive_data:
        # 第一阶段：描述性编辑
        print("执行描述性编辑阶段...")
        _, edited_model, _ = editor.edit(
            prompts=descriptive_prompts,
            target_new=descriptive_targets,
            subject=subjects,
            keep_original_weight=True,
            test_generation=False,  # 不进行评测
        )
        
        editor.model = edited_model
        
    # 第二阶段：基本编辑并评测
    print("执行基本编辑阶段并评测...")
    metrics, final_model, _ = editor.edit(
        prompts=basic_prompts,
        target_new=basic_targets,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        pre_file=args.pre_file,
        pre_edit=pre_edit,
        test_generation=True,  # 执行评测
        train_ds=train_ds
    )
    

    # 保存和评估结果
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_results.json')
    json.dump(metrics, open(result_path, 'w'), indent=4)
    eval(result_path)
