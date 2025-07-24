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
        if ' ' not in i['original_prediction']:
            i['original_prediction'] += ' '
        if dataset_type in ['unke','counterfact','wikiupdate','mquake'] and ' ' not in i['para_prediction']:
            i['para_prediction'] += ' '
        if 'sub_pred' in i:
            for j in range(len(i['sub_pred'])):
                if ' ' not in i['sub_pred'][j]:
                    i['sub_pred'][j] += ' '

    matrics = {}
    
    print("Calculating metrics for original questions...")
    bleu_scores = []
    rouge1s = []
    rouge2s = []
    rougels = []
    
    bleu_scores_para = []
    rouge1s_para = []
    rouge2s_para = []
    rougels_para = []
    rougels_sub = []
    rouge1s_sub = []
    rouge2s_sub = []
    rouge = Rouge()

    for index in tqdm(range(len(data)), desc='Calculate BLEU&ROUGE Score'):
        score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
        bleu_scores.append(score)
        
        if dataset_type in ['unke','counterfact','wikiupdate','mquake']:
            score = sentence_bleu([data[index]['answer']], data[index]['para_prediction'])
            bleu_scores_para.append(score)
            
        scores = rouge.get_scores(data[index]['original_prediction'], data[index]['answer'])
        rouge1s.append(scores[0]['rouge-1']['r'])
        rouge2s.append(scores[0]['rouge-2']['r'])
        rougels.append(scores[0]['rouge-l']['r'])
        
        if dataset_type in ['unke','counterfact','wikiupdate','mquake']:
            scores = rouge.get_scores(data[index]['para_prediction'], data[index]['answer'])
            rouge1s_para.append(scores[0]['rouge-1']['r'])
            rouge2s_para.append(scores[0]['rouge-2']['r'])
            rougels_para.append(scores[0]['rouge-l']['r'])
            
        sub_ls = 0
        sub_1s = 0
        sub_2s = 0
        for i in range(len(data[index]['sub_pred'])):
            scores = rouge.get_scores(data[index]['sub_pred'][i], data[index]['sub_answer'][i])
            sub_1s += scores[0]['rouge-1']['r']
            sub_2s += scores[0]['rouge-2']['r']
            sub_ls += scores[0]['rouge-l']['r']
        rouge1s_sub.append(sub_1s/len(data[index]['sub_pred']))
        rouge2s_sub.append(sub_2s/len(data[index]['sub_pred']))
        rougels_sub.append(sub_ls/len(data[index]['sub_pred']))   

    temp_original = {}
    temp_para = {}
    temp_sub = {}
    
    temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores)
    temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s)
    temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s)
    temp_original['ROUGE-L'] = sum(rougels) / len(rougels)
    
    if dataset_type in ['unke','cf','wikiupdate','mquake']:
        temp_para['BLEU SCORE'] = sum(bleu_scores_para) / len(bleu_scores_para)
        temp_para['ROUGE-1'] = sum(rouge1s_para) / len(rouge1s_para)
        temp_para['ROUGE-2'] = sum(rouge2s_para) / len(rouge2s_para)
        temp_para['ROUGE-L'] = sum(rougels_para) / len(rougels_para)

    temp_sub['ROUGE-1'] = sum(rouge1s_sub) / len(rouge1s_sub)
    temp_sub['ROUGE-2'] = sum(rouge2s_sub) / len(rouge2s_sub)
    temp_sub['ROUGE-L'] = sum(rougels_sub) / len(rougels_sub)
    
    print("***********Calculate BERT Similarity Score**************")
    sentences1 = [i['answer'] for i in data]
    sentences2 = [i['original_prediction'] for i in data]
    if dataset_type in ['unke','cf','wikiupdate','mquake']:
        sentences3 = [i['para_prediction'] for i in data]
    model = SentenceTransformer(model_path, device=f"cuda:{device}")

    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    if dataset_type in ['unke','cf','wikiupdate','mquake']:
        embeddings3 = model.encode(sentences3, convert_to_tensor=True, show_progress_bar=True)
        
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()

    if dataset_type in ['unke','cf','wikiupdate','mquake']:
        cosine_scores = util.cos_sim(embeddings1, embeddings3)
        temp_para['Bert Score'] = cosine_scores.diagonal().mean().item()
        
    matrics['Original'] = temp_original
    if dataset_type in ['unke','cf','wikiupdate','mquake']:
        matrics['Para'] = temp_para
    matrics['Sub'] = temp_sub
    
    print("***********Result**************")
    print(matrics)
    
    return matrics

