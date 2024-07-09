# coding=utf-8
import json
import sys
import os
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
sys.setrecursionlimit(2000)

class DatasizeError(Exception):
    def __init__(self, message) :
        super().__init__(message)
        self.message=message

class SampleError(Exception):
    def __init__(self, message) :
        super().__init__(message)
        self.message=message

class CaseidError(Exception):
    def __init__(self, message) :
        super().__init__(message)
        self.message=message

error_msg={
    1: "Wrong data size",
    2: "Wrong sample format",
    3: "Wrong case id"
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(score, out_p):
    result = dict()
    result['success']=True
    total_score = score['Edit_acc']['final_score'] * 0.2 + score['portability']['final_score'] * 0.35 + score['locality']['final_score']  * 0.35 + score['fluency'] * 0.1
    result['score'] = total_score
    result['scoreJson'] = {'score': total_score, 'Edit_acc':score['Edit_acc']['final_score'], 'portability':score['portability']['final_score'], 'locality':score['locality']['final_score'], 'fluency':score['fluency']}
    dump_2_json(result,out_p)

def sample_format(sample_list):
    tag=True
    for x in sample_list:                                                          
        list1 = x.keys()
        list2 = x['pre'].keys()
        list3 = x['requested_rewrite'].keys()
        list4 = x['post'].keys()
        if(list(list1)!=['pre', 'case_id', 'requested_rewrite', 'post']):
            tag=False
            break
        elif(list(list2)!=['rewrite_ans','rephrase_ans','locality_ans','portability_ans'] and list(list2)!=['rewrite_ans','rephrase_ans','portability_ans']):
            tag=False
            break
        elif(list(list3)!=['prompt', 'target_new', 'ground_truth', 'portability', 'locality', 'subject','rephrase_prompt']):
            tag=False
            break
        elif(list(list4)!=['rewrite_ans','rephrase_ans','locality_ans','portability_ans','fluency'] and list(list4)!=['rewrite_ans','rephrase_ans','portability_ans','fluency']):
            tag=False
            break  
    return tag

def test_case_id(sample_list):
    tag =True
    for x in range(len(sample_list)-1):
        if(sample_list[x+1]['case_id']!=sample_list[x]['case_id']+1):
            tag = False
            break
    return tag

def check_format(submit_p):
    with open(submit_p, 'r',encoding='utf-8') as file:
        submit_file=json.load(file)
    if len(submit_file)<3:
        raise DatasizeError("Wrong data size")
    if (not sample_format(submit_file)):
        raise SampleError("Wrong sample format")
    if (not test_case_id(submit_file)):
        raise CaseidError("Wrong case id")

def compute_acc(answers,outputs):
    model_path = './paraphrase-multilingual-MiniLM-L12-v2'
    bleu_scores = []
    rouge1s=[]
    rouge2s=[]
    rougels=[]
    rouge = Rouge()
    for an,ou in zip(answers,outputs):
        score = sentence_bleu([an], ou)
        bleu_scores.append(score)
        scores = rouge.get_scores(ou,an)
        rouge1s.append(scores[0]['rouge-1']['r'])
        rouge2s.append(scores[0]['rouge-2']['r'])
        rougels.append(scores[0]['rouge-l']['r'])

    temp_metrics = {}
    temp_metrics['BLEU SCORE'] = sum(bleu_scores) / len(bleu_scores)
    temp_metrics['ROUGE-1'] = sum(rouge1s) / len(rouge1s)
    temp_metrics['ROUGE-2'] = sum(rouge2s) / len(rouge2s)
    temp_metrics['ROUGE-L'] = sum(rougels) / len(rougels)

    model = SentenceTransformer(model_path, device="cpu")

    embeddings1 = model.encode(answers, convert_to_tensor=True)
    embeddings2 = model.encode(outputs, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    temp_metrics['Bert Score'] = cosine_scores.diagonal().mean().item()
    temp_metrics['final_score'] = (temp_metrics['ROUGE-L']+temp_metrics['Bert Score'])/2
    temp_metrics['final_score'] = temp_metrics['final_score']*100
    
    return temp_metrics

def eval_score(result_path):
    with open(result_path, 'r', encoding='utf-8') as file:
        data=json.load(file)
    metrics = {}

    #evaluate Edit_acc
    rewrite_answer = [i['requested_rewrite']['target_new'] for i in data]
    rewrite_outputs = [i['post']['rewrite_ans'] for i in data]
    metrics['Edit_acc'] = compute_acc(rewrite_answer,rewrite_outputs)

    #evaluate portability
    portability_answer = []
    portability_outputs = []
    for item in data:
        for an in item['requested_rewrite']['portability']['por_hop']['ground_truth']:
            portability_answer.append(an)
        for ou in item['post']['portability_ans']:
            portability_outputs.append(ou)
    metrics['portability'] = compute_acc(portability_answer,portability_outputs)

    #evaluate locality
    locality_answer = []
    locality_outputs = []
    for item in data:
        if ('locality_ans' not in item['post'].keys() or len(item['requested_rewrite']['locality']['loc_hop']['prompt'])==0):
            continue
        for an in item['requested_rewrite']['locality']['loc_hop']['ground_truth']:
            locality_answer.append(an)
        for ou in item['post']['locality_ans']:
            locality_outputs.append(ou)
    metrics['locality'] = compute_acc(locality_answer,locality_outputs)

     #evaluate fluency
    fluencys = [i['post']['fluency']['ngram_entropy'] for i in data]
    metrics['fluency'] = sum(fluencys) / len(fluencys) *10

    return metrics

if __name__=="__main__":
    
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]

    # read submit and answer file from first parameter
    with open(in_param_path, 'r', encoding='utf-8') as load_f:
        input_params = json.load(load_f)

    # 选手提交的结果文件路径
    submit_path=input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_path)

    try:
        check_format(submit_path)
        score = eval_score(submit_path)
        report_score(score, out_path)
    except DatasizeError as e:
        check_code = 1
        report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
    except SampleError as e:
        check_code = 2
        report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
    except CaseidError as e:
        check_code = 3
        report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
