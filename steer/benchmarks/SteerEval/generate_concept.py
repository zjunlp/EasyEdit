import argparse
import json
import os
import re
from openai import OpenAI
from prompts import PROMPTS

MODEL = 'DeepSeek-V3.2'
BASE_URL="xxxx"
API_KEY="xxxx"

def ask_llm(content: str, model: str) -> str:
    completion = client.chat.completions.create(
        model=model, 
        messages = [
            {"role": "user", "content": content},
        ],
        
    )
    return completion.choices[0].message.content

def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    s = m.group(1) if m else text.strip()
    try:
        return json.loads(s)
    except :
         raise Exception(f"JSON Format Error")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate question dataset.")
    parser.add_argument('-f','--file', default= 'example',      type=str, help="Path to the input file.")
    parser.add_argument('--domain', nargs='+', default=['sentiment', 'personality', 'reasoning patterns',  'language features'], help="Domains to generate concepts for.")
    parser.add_argument('-n1','--n1', type=int, default=1, help="Number of concepts to generate of Level 1.")
    parser.add_argument('-n2','--n2', type=int, default=1, help="Number of concepts to generate of Level 2.")
    parser.add_argument('-n3','--n3', type=int, default=1, help="Number of concepts to generate of Level 3.")
    
    
    args = parser.parse_args()
    MAX_RETRIES = 5
    client = OpenAI(   
        base_url=BASE_URL,   
        api_key=API_KEY , 
    )

    path = f"data/{args.file}"
    
    domain = args.domain
    
    for d in domain:
        print(f"\033[94mDomain: {d}\033[0m")
        prompt_expand_domain = PROMPTS["expand_domain_stage_1"].format(USER_INPUT_DOMAIN=d)
        
        domain_description = ask_llm(prompt_expand_domain, MODEL)
        print(f"Domain Specification:\n{domain_description}")

        prompt_generate_concept = PROMPTS["generate_concepts_stage_2"].format(
            DOMAIN_NAME=d,
            DOMAIN_DESCRIPTION = domain_description,
            N_L1=args.n1,
            N_L2=args.n2,
            N_L3=args.n3)

        concepts = ask_llm(prompt_generate_concept, MODEL)
        print(concepts)
        print('\n\n')
        result = extract_json(concepts)
        result['domain_description'] = domain_description
        if not os.path.exists(f'{path}/{d.replace(" ", "_")}'):
            os.makedirs(f'{path}/{d.replace(" ", "_")}')
        with open(f'{path}/{d.replace(" ", "_")}/concepts.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    for d in domain:
        data = json.load(open(f'{path}/{d.replace(" ", "_")}/concepts.json', 'r', encoding='utf-8'))
        print(f"\033[94mDomain: {d}\033[0m")
        all_concept = { 'L1':[], 'L2':[], 'L3':[]}
        for l1 in data['L1_concepts']:
            # print(f"L1: {l1['concept']}")
            all_concept['L1'].append({
                'concept_id': f"L1_{len(all_concept['L1'])+1}",
                'concept': l1['concept'],
                'level': 1
            })
            for l2 in l1['L2_subconcepts']:
                # print(f"  L2: {l2['concept']}")
                all_concept['L2'].append({
                    'concept_id': f"L2_{len(all_concept['L2'])+1}",
                    'concept': l2['concept'],
                    'level': 2
                })
                for l3 in l2['L3_features']:
                    # print(f"    L3: {l3['concept']}")
                    all_concept['L3'].append({
                        'concept_id': f"L3_{len(all_concept['L3'])+1}",
                        'concept': l3['concept'],
                        'level': 3
                    })
        total = []
        for level, concepts in all_concept.items():
            print(f"\033[92mLevel: {level}, Count: {len(concepts)}\033[0m")
            total.extend(concepts)
            with open(f'{path}/{d.replace(" ", "_")}/concepts_{level}.json', 'w', encoding='utf-8') as f:
                json.dump(concepts, f, indent=2, ensure_ascii=False)
        with open(f'{path}/{d.replace(" ", "_")}/concepts_all.json', 'w', encoding='utf-8') as f:
            json.dump(total, f, indent=2, ensure_ascii=False)
        print('\n\n')   
