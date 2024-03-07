import json
import argparse
import os

data = json.load(open('./data/concept_data.json'))

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="FT", type=str)
parser.add_argument("--model", default="llama2chat",type=str)
parser.add_argument("--module",default="intra",type=str)
args = parser.parse_args()


def process_str(s):
    s = s.lstrip().replace('\n','')
    while s.startswith(':') or s.startswith('\"'):
        s = s[1:]
    if '.' in s:
        first_period_index = s.find('.')
        if s[first_period_index-1].isdigit():
            next_period_index = s.find('.', first_period_index + 20)
            return s[first_period_index +1 :next_period_index]
        return s[:first_period_index+1]
    else:
        return s




str_temp ='''Prediction sentence: [PREDICTION]

Sentence A: [TARGET].
Sentence B: [GROUND].

Check the prediction sentence and Give a score from -1 to 1:
Score 1: close meaning to sentence A
Score 0: neither relevant to A nor B
Score -1: close meaning to sentence B

Output format is {Score:{}, Reason:{}}
'''

print(str_temp)


out_dir = 'trans_check'
if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)
result_dir = "final_result_upload"
with open(f"{result_dir}/{args.method}_results_{args.model}_{args.module}.json", "r") as f:
    result = json.load(f)
    
outputs = []
result = [i['post']['gen_concept_text'] for i in result]
for id , i in enumerate(data):
    item = {}
    item['id'] = id
    item['label'] = i['concept_name']
    target_str = i[f'module_{args.module}']['replace_def']
    ground_str = i['concept_def']
    processed_str = process_str(result[id])

    input_str = str_temp.replace('[PREDICTION]',processed_str).replace('[TARGET]',target_str).replace('[GROUND]',ground_str)
    item['input_str'] = input_str
    outputs.append(item)

json.dump(outputs,open(f'{out_dir}/{args.method}_check_{args.model}_{args.module}.json','w'),indent=4)



import random
a = random.sample(outputs,5)
a = [i['input_str'] for i in a]
for t in a:
    print(t)
    print('='*40)
     