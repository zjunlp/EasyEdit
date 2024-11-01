import zhipuai
from zhipuai import ZhipuAI
import torch
from transformers import AutoTokenizer, AutoModel
import string
import yaml
import logging
import httpx

    
with open('EasyEdit/hparams/DeepEdit_Api/chatglm_api.yaml', 'r') as file:
    config = yaml.safe_load(file)
# read api key
api = config.get('api_key', None)
# read prompts
prompts_dir = config.get('prompts_dir', None)
with open(prompts_dir + '/multihop-cot-prompts.txt', 'r') as f:
    task_prompt = f.read()
with open(prompts_dir + '/delete-prompt.txt', 'r') as f:
    delete_prompt = f.read()
with open(prompts_dir + '/conflict-prompt.txt', 'r') as f:
    conflict_prompt = f.read()
with open(prompts_dir + '/conflict-prompt-1.txt', 'r') as f:
    conflict_prompt_1 = f.read()
with open(prompts_dir + '/conflict-prompt-2.txt', 'r') as f:
    conflict_prompt_2 = f.read()
with open(prompts_dir + '/entity-prompt.txt', 'r') as f:
    entity_prompt = f.read()
with open(prompts_dir + '/exist-prompt.txt', 'r') as f:
    exist_prompt = f.read()

contriver_dir = config.get('contriver_dir', None)
contriever = AutoModel.from_pretrained(contriver_dir).cuda()
tokenizer_dir = config.get('tokenizer_dir', None)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

logging.getLogger("httpx").setLevel(logging.WARNING)
def call_glm(cur_prompt, stop, temperature=0):
    client = ZhipuAI(api_key=api)
    response = client.chat.completions.create(
        model="glm-4-plus",
        max_tokens=256,
        stop = stop,
        messages=[
            {
                "role":"user",
                "content":cur_prompt
            }
        ],
        temperature = temperature
    )
    return response.choices[0].message.content

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in range(0, len(sents), BSZ):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=2):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(min(len(fact_embs), k), largest=True)
    return knn.indices

def searchFact(sentence_1, sentence_2, fact_ls):
    embs = get_sent_embeddings(fact_ls, contriever, tokenizer)
    
    prompt_1 = delete_prompt + "\n\nSentence A: " + sentence_1 + '\nSentence B: ' + sentence_2 + '\nSentence B after removal: '
    gen = call_glm(prompt_1, ['.\n']).strip()
    thought_for_search = gen

    fact_ids = retrieve_facts(thought_for_search, embs, contriever, tokenizer)
    fact_sent_ls = [fact_ls[id_] + '.' for id_ in fact_ids]
    return fact_sent_ls

def newEntity(sentence_1, sentence_2):
    prompt_1 = entity_prompt + "\n\nSentence A: " + sentence_1 + '\nSentence B: ' + sentence_2 + '\nAnswer: '
    gen = call_glm(prompt_1, ['.\n']).strip()
    return gen

def entityExist(entity, sentence):
    if entity.lower() in sentence.lower():
        return True
    prompt_4 = exist_prompt + "\n\nSentence: " + sentence + '\nEntity: ' + entity + '\nAnswer: '
    gen = call_glm(prompt_4, ['.\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_glm(prompt_4, ['.\n'], temperature = 1).strip().strip(string.punctuation)
    
    if gen.lower() == 'yes':
        return True
    else:
        return False
    
def judgeConflict1(sentence_1, sentence_2):
    prompt_2 = conflict_prompt_1 + "\n\nSentence A: " + sentence_1 + '\nSentence B: ' + sentence_2 + '\nAnswer: '
    gen = call_glm(prompt_2, ['.\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        
        gen = call_glm(prompt_2, ['.\n'], temperature = 1).strip().strip(string.punctuation)
    
    if gen.lower() == 'yes':
        return True
    else:
        return False
    
def judgeConflict2(sentence_1, sentence_2):
    prompt_2 = conflict_prompt_2 + "\n\nSentence A: " + sentence_1 + '\nSentence B: ' + sentence_2 + '\nAnswer: '
    gen = call_glm(prompt_2, ['.\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        
        gen = call_glm(prompt_2, ['.\n'], temperature = 1).strip().strip(string.punctuation)
    
    if gen.lower() == 'yes':
        return True
    else:
        return False
    
def judgeConflict(sentence_1, sentence_2):
    prompt_2 = conflict_prompt + "\n\nSentence A: " + sentence_1 + '\nSentence B: ' + sentence_2 + '\nAnswer: '
    gen = call_glm(prompt_2, ['.\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_glm(prompt_2, ['.\n'], temperature = 1).strip().strip(string.punctuation)
    
    if gen.lower() == 'yes':
        return True
    else:
        return False
    
def oneStep(prompt, last_entity, new_entity, last_thought, count, fact_ls):
    gen = call_glm(prompt, ['.#', '.\n']).strip()
    if 'the answer' in gen or 'the answers' in gen:
        return new_entity
    current_thought = (gen + '.').strip()
    fact_sent_ls = searchFact(last_thought, current_thought, fact_ls)

    output_ls = []
    for fact_sent_ in fact_sent_ls[::-1]:
        last_entity_flag = entityExist(last_entity, fact_sent_)
        exist_flag = entityExist(new_entity, fact_sent_)
        conflict_flag = judgeConflict(fact_sent_, current_thought) and judgeConflict1(fact_sent_, current_thought) and judgeConflict2(fact_sent_, current_thought)
        if (exist_flag and not last_entity_flag) or conflict_flag:
            new_thought = fact_sent_
            output = {
                'last_entity' : new_entity,
                'new_entity' : newEntity(last_thought, new_thought),
                'prompt' : prompt + new_thought + '# ',
                'last_thought' : new_thought,
                'count' : count + 1,
            }
            output_ls.append(output)
            
    if len(output_ls) == 0:
        new_thought = current_thought
        output = {
            'last_entity' : new_entity,
            'new_entity' : newEntity(last_thought, new_thought),
            'prompt' : prompt + new_thought + '# ',
            'last_thought' : new_thought,
            'count' : count + 1,
        }
        output_ls.append(output)

    return output_ls
