import json
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse
import sys
import os
sys.setrecursionlimit(2000)
def calculate_metrics(data):
    for i in data:
        if ' ' not in i['original_prediction']:
            i['original_prediction'] += ' '
    bleu_scores = []
    rouge1s = []
    rouge2s = []
    rougels = []
    rouge = Rouge()
    for index in tqdm(range(len(data)), desc='Calculate BLEU&ROUGE Score'):
        score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
        bleu_scores.append(score)
        scores = rouge.get_scores(data[index]['original_prediction'], data[index]['answer'])
        rouge1s.append(scores[0]['rouge-1']['r'])
        rouge2s.append(scores[0]['rouge-2']['r'])
        rougels.append(scores[0]['rouge-l']['r'])
    temp_original = {}
    temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s) if rouge1s else 0
    temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s) if rouge2s else 0
    temp_original['ROUGE-L'] = sum(rougels) / len(rougels) if rougels else 0

    # cal bert score
    print("***********Calculate BERT Similarity Score**************")
    sentences1 = [i['answer'] for i in data]
    sentences2 = [i['original_prediction'] for i in data]
    model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # print(cosine_scores.shape)
    temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()
    return temp_original
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='output/unke_Llama3-8B-Instruct_cf_result.json', type=str)
    parser.add_argument('--model_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()

    ds_name =  args.file_path.split('_')[-2]
    with open(args.file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    # data = [i for i in data if 'sub_pred' in i.keys()]
    

    if ds_name == "editevery":
        category_data = {}
        for item in data:
            category = item['category']
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(item)

        matrics = {}
        for category, cat_data in category_data.items():
            print(f"Calculating metrics for category: {category}")
            metrics = calculate_metrics(cat_data)
            matrics[category] = metrics

        print("***********Result**************")
        print(matrics)
    else:
        for i in data:
            if ' ' not in i['original_prediction']:
                i['original_prediction'] += ' '
            if ds_name in ['unke','cf'] and ' ' not in i['para_prediction']:
                i['para_prediction'] += ' '
            for j in range(len(i['sub_pred'])):
                if ' ' not in i['sub_pred'][j]:
                    i['sub_pred'][j] += ' '

            # if i['original_prediction'] == '':
            #     i['original_prediction'] = ' '
            # if i['para_prediction'] == '':
            #     i['para_prediction'] = ' '
        matrics = {}

        # cal bleu
        bleu_scores = []
        rouge1s=[]
        rouge2s=[]
        rougels=[]
        
        bleu_scores_para = []
        rouge1s_para=[]
        rouge2s_para=[]
        rougels_para=[]
        rougels_sub=[]
        rouge1s_sub=[]
        rouge2s_sub=[]
        rouge = Rouge()

        for index in tqdm(range(len(data)),desc='Calculate BLEU&ROUGE Score'):
            score = sentence_bleu([data[index]['answer']], data[index]['original_prediction'])
            bleu_scores.append(score)
            if ds_name in ['unke','cf']:
                score = sentence_bleu([data[index]['answer']], data[index]['para_prediction'])
                bleu_scores_para.append(score)
            scores = rouge.get_scores(data[index]['original_prediction'],data[index]['answer'])
            rouge1s.append(scores[0]['rouge-1']['r'])
            rouge2s.append(scores[0]['rouge-2']['r'])
            rougels.append(scores[0]['rouge-l']['r'])
            if ds_name in ['unke','cf']:
                scores = rouge.get_scores(data[index]['para_prediction'],data[index]['answer'])
                rouge1s_para.append(scores[0]['rouge-1']['r'])
                rouge2s_para.append(scores[0]['rouge-2']['r'])
                rougels_para.append(scores[0]['rouge-l']['r'])
            sub_ls = 0
            sub_1s = 0
            sub_2s = 0
            for i in range(len(data[index]['sub_pred'])):
                scores = rouge.get_scores(data[index]['sub_pred'][i],data[index]['sub_answer'][i])
                sub_1s += scores[0]['rouge-1']['r']
                sub_2s += scores[0]['rouge-2']['r']
                sub_ls += scores[0]['rouge-l']['r']
            rouge1s_sub.append(sub_1s/len(data[index]['sub_pred']))
            rouge2s_sub.append(sub_2s/len(data[index]['sub_pred']))
            rougels_sub.append(sub_ls/len(data[index]['sub_pred']))   

        
        temp_original = {}
        temp_para = {}
        temp_sub={}
        temp_original['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores)
        temp_original['ROUGE-1'] = sum(rouge1s) / len(rouge1s)
        temp_original['ROUGE-2'] = sum(rouge2s) / len(rouge2s)
        temp_original['ROUGE-L'] = sum(rougels) / len(rougels)
        if ds_name in ['unke','cf']:
            temp_para['BLEU SCORE'] = sum(bleu_scores_para) / len(bleu_scores_para)
            temp_para['ROUGE-1'] = sum(rouge1s_para) / len(rouge1s_para)
            temp_para['ROUGE-2'] = sum(rouge2s_para) / len(rouge2s_para)
            temp_para['ROUGE-L'] = sum(rougels_para) / len(rougels_para)

        temp_sub['ROUGE-1'] = sum(rouge1s_sub) / len(rouge1s_sub)
        temp_sub['ROUGE-2'] = sum(rouge2s_sub) / len(rouge2s_sub)
        temp_sub['ROUGE-L'] = sum(rougels_sub) / len(rougels_sub)
        # cal bert score
        print("***********Calculate BERT Similarity Score**************")
        sentences1 = [i['answer'] for i in data]
        sentences2 = [i['original_prediction'] for i in data]
        if ds_name in ['unke','cf']:
            sentences3 = [i['para_prediction'] for i in data]
        model = SentenceTransformer(args.model_path, device=f"cuda:{args.device}")

        embeddings1 = model.encode(sentences1, convert_to_tensor=True,show_progress_bar=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True,show_progress_bar=True)
        if ds_name in ['unke','cf']:
            embeddings3 = model.encode(sentences3, convert_to_tensor=True,show_progress_bar=True)
        # Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # print(cosine_scores.shape)
        temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()
        temp_bert_score = cosine_scores.diagonal().cpu().numpy().tolist()

        if ds_name in ['unke','cf']:
            cosine_scores = util.cos_sim(embeddings1, embeddings3)
            # print(cosine_scores.shape)
            temp_para['Bert Score'] = cosine_scores.diagonal().mean().item()
            temp_bert_score_para = cosine_scores.diagonal().cpu().numpy().tolist()
        matrics['Original']=temp_original
        if ds_name in ['unke','cf']:
            matrics['Para']=temp_para
        matrics['Sub']=temp_sub
        # temp_result = [bleu_scores,bleu_scores_para,rouge1s,rouge1s_para,rouge2s,rouge2s_para,rougels,rougels_para,temp_bert_score,temp_bert_score_para]
        # with open('data_memit.json', 'w') as file:
        #     json.dump(temp_result, file)
        print("***********Result**************")
        print(matrics)

