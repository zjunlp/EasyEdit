import json
import os.path as path
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def eval_akew_unstructured(result_path, dataset_type, model_path='all-MiniLM-L6-v2', device=4):
    if not path.exists(result_path):
        print(f"Result file not found: {result_path}")
        return
        
    print(f"Evaluating unstructured results from: {result_path}")
    
    with open(result_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    print(f"Loaded {len(data)} samples for evaluation")
    for i in data:
        if ' ' not in i.get('original_prediction', ''):
            i['original_prediction'] = i.get('original_prediction', '') + ' '
        if dataset_type in ['unke', 'counterfact', 'mquake', 'wikiupdate']:
            if ' ' not in i.get('para_prediction', ''):
                i['para_prediction'] = i.get('para_prediction', '') + ' '
        if 'sub_pred' in i and i['sub_pred']:
            for j in range(len(i['sub_pred'])):
                if ' ' not in i['sub_pred'][j]:
                    i['sub_pred'][j] += ' '

    metrics = {}
    rouge = Rouge()
    
    print("Calculating metrics for original questions...")
    bleu_scores = []
    rouge1s = []
    rouge2s = []
    rougels = []
    
    for index in tqdm(range(len(data)), desc='Calculate Original Question Metrics'):
        # BLEU Score
        try:
            score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
        
        # ROUGE Scores
        try:
            scores = rouge.get_scores(data[index]['original_prediction'], data[index]['answer'])
            rouge1s.append(scores[0]['rouge-1']['r'])
            rouge2s.append(scores[0]['rouge-2']['r'])
            rougels.append(scores[0]['rouge-l']['r'])
        except:
            rouge1s.append(0.0)
            rouge2s.append(0.0)
            rougels.append(0.0)
    
    temp_original = {
        'BLEU Score': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        'ROUGE-1': sum(rouge1s) / len(rouge1s) if rouge1s else 0,
        'ROUGE-2': sum(rouge2s) / len(rouge2s) if rouge2s else 0,
        'ROUGE-L': sum(rougels) / len(rougels) if rougels else 0
    }
    
    temp_para = {}
    if dataset_type in ['unke', 'counterfact', 'mquake', 'wikiupdate']:
        print("Calculating metrics for paraphrase questions...")
        bleu_scores_para = []
        rouge1s_para = []
        rouge2s_para = []
        rougels_para = []
        
        for index in tqdm(range(len(data)), desc='Calculate Paraphrase Question Metrics'):
            # BLEU Score
            try:
                score = sentence_bleu([data[index]['answer']], data[index]['para_prediction'])
                bleu_scores_para.append(score)
            except:
                bleu_scores_para.append(0.0)
            
            # ROUGE Scores
            try:
                scores = rouge.get_scores(data[index]['para_prediction'], data[index]['answer'])
                rouge1s_para.append(scores[0]['rouge-1']['r'])
                rouge2s_para.append(scores[0]['rouge-2']['r'])
                rougels_para.append(scores[0]['rouge-l']['r'])
            except:
                rouge1s_para.append(0.0)
                rouge2s_para.append(0.0)
                rougels_para.append(0.0)
        
        temp_para = {
            'BLEU Score': sum(bleu_scores_para) / len(bleu_scores_para) if bleu_scores_para else 0,
            'ROUGE-1': sum(rouge1s_para) / len(rouge1s_para) if rouge1s_para else 0,
            'ROUGE-2': sum(rouge2s_para) / len(rouge2s_para) if rouge2s_para else 0,
            'ROUGE-L': sum(rougels_para) / len(rougels_para) if rougels_para else 0
        }
    
    print("Calculating metrics for sub questions...")
    rouge1s_sub = []
    rouge2s_sub = []
    rougels_sub = []
    
    for index in tqdm(range(len(data)), desc='Calculate Sub Questions Metrics'):
        if 'sub_pred' in data[index] and 'sub_answer' in data[index]:
            if data[index]['sub_pred'] and data[index]['sub_answer']:
                sub_r1 = 0
                sub_r2 = 0
                sub_rl = 0
                valid_count = 0
                
                min_len = min(len(data[index]['sub_pred']), len(data[index]['sub_answer']))
                for i in range(min_len):
                    try:
                        scores = rouge.get_scores(data[index]['sub_pred'][i], data[index]['sub_answer'][i])
                        sub_r1 += scores[0]['rouge-1']['r']
                        sub_r2 += scores[0]['rouge-2']['r']
                        sub_rl += scores[0]['rouge-l']['r']
                        valid_count += 1
                    except:
                        continue
                
                if valid_count > 0:
                    rouge1s_sub.append(sub_r1 / valid_count)
                    rouge2s_sub.append(sub_r2 / valid_count)
                    rougels_sub.append(sub_rl / valid_count)
                else:
                    rouge1s_sub.append(0.0)
                    rouge2s_sub.append(0.0)
                    rougels_sub.append(0.0)
            else:
                rouge1s_sub.append(0.0)
                rouge2s_sub.append(0.0)
                rougels_sub.append(0.0)
        else:
            rouge1s_sub.append(0.0)
            rouge2s_sub.append(0.0)
            rougels_sub.append(0.0)
    
    temp_sub = {
        'ROUGE-1': sum(rouge1s_sub) / len(rouge1s_sub) if rouge1s_sub else 0,
        'ROUGE-2': sum(rouge2s_sub) / len(rouge2s_sub) if rouge2s_sub else 0,
        'ROUGE-L': sum(rougels_sub) / len(rougels_sub) if rougels_sub else 0
    }
    
    print("***********Calculate BERT Similarity Score**************")
    try:
        model = SentenceTransformer(model_path, device=f"cuda:{device}")
        
        # Original questions BERT Score
        sentences1 = [i['answer'] for i in data]
        sentences2 = [i['original_prediction'] for i in data]
        
        embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
        
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        temp_original['BERT Score'] = cosine_scores.diagonal().mean().item()
        
        # Paraphrase questions BERT Score
        if dataset_type in ['unke', 'counterfact', 'mquake', 'wikiupdate']:
            sentences3 = [i['para_prediction'] for i in data]
            embeddings3 = model.encode(sentences3, convert_to_tensor=True, show_progress_bar=True)
            cosine_scores_para = util.cos_sim(embeddings1, embeddings3)
            temp_para['BERT Score'] = cosine_scores_para.diagonal().mean().item()
            
    except Exception as e:
        print(f"Error calculating BERT Score: {e}")
        temp_original['BERT Score'] = 0.0
        if temp_para:
            temp_para['BERT Score'] = 0.0
    
    metrics['Original'] = temp_original
    if temp_para:
        metrics['Paraphrase'] = temp_para
    metrics['Sub Questions'] = temp_sub
    
    print("\n" + "="*60)
    print("           AKEW UNSTRUCTURED EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nDataset: {dataset_type.upper()}")
    print(f"Total Samples: {len(data)}")
    
    print(f"\n ORIGINAL QUESTION METRICS:")
    for metric, value in temp_original.items():
        print(f"   {metric:<12}: {value:.4f}")
    
    if temp_para:
        print(f"\n PARAPHRASE QUESTION METRICS:")
        for metric, value in temp_para.items():
            print(f"   {metric:<12}: {value:.4f}")
    
    print(f"\n SUB QUESTIONS METRICS:")
    for metric, value in temp_sub.items():
        print(f"   {metric:<12}: {value:.4f}")
    
    print("\n" + "="*60)
    
    return metrics


def eval_akew_unstructured_by_category(result_path, dataset_type, model_path='sentence-transformers/all-MiniLM-L6-v2', device=0):
    if not path.exists(result_path):
        print(f"Result file not found: {result_path}")
        return
        
    with open(result_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if dataset_type != "editevery":
        print("Category evaluation is only for editevery dataset")
        return eval_akew_unstructured(result_path, dataset_type, model_path, device)
    
    category_data = {}
    for item in data:
        category = item.get('category', 'unknown')
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(item)
    
    print(f"Found {len(category_data)} categories: {list(category_data.keys())}")
    
    all_metrics = {}
    for category, cat_data in category_data.items():
        print(f"\nCalculating metrics for category: {category} ({len(cat_data)} samples)")
        
        temp_file = f"temp_{category}_data.json"
        with open(temp_file, 'w') as f:
            json.dump(cat_data, f)
        
        category_metrics = eval_akew_unstructured(temp_file, dataset_type, model_path, device)
        all_metrics[category] = category_metrics
        
        # 清理临时文件
        os.remove(temp_file)
    
    print("\n" + "="*80)
    print("           EDITEVERY CATEGORY-WISE RESULTS")
    print("="*80)
    
    for category, metrics in all_metrics.items():
        print(f"\n  CATEGORY: {category.upper()}")
        print("-" * 40)
        if 'Original' in metrics:
            for metric, value in metrics['Original'].items():
                print(f"   {metric:<12}: {value:.4f}")
    
    return all_metrics