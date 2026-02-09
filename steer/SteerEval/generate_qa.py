import asyncio
import json
import os
import re
import argparse
from openai import AsyncOpenAI, APIError, RateLimitError
from tqdm.asyncio import tqdm_asyncio
from prompts import PROMPTS  

MODEL = 'gpt-4.1-mini'
BASE_URL="xxxx"
API_KEY="xxxx"

MAX_RETRIES = 5
CONCURRENCY_LIMIT = 40  


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

class JsonParseError(Exception):
    pass


def read_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_file(file_path: str, content):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if isinstance(content, (dict, list)):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(content))

def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    s = m.group(1) if m else text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
         raise JsonParseError(f"JSON Format Error: {e.msg}")

def extract_xml(text: str) -> dict:
    tags = ["rationale", "refined_question", "positive_answer", "negative_answer", "metadata"]
    result = {}
    for tag in tags:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[tag] = match.group(1).strip() 
    return result



async def ask_llm_async(content: str, model: str) -> str:

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content

async def query_async(content: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore: 
        for i in range(MAX_RETRIES):
            try:
                resp = await ask_llm_async(content, MODEL)
                return resp
            except RateLimitError:
                print(f"Rate limit hit. Retrying ({i+1}/{MAX_RETRIES})... Waiting 5s.")
                await asyncio.sleep(5) 
            except Exception as e:
                print(f"Error: {e}. Retrying ({i+1}/{MAX_RETRIES})...")
                await asyncio.sleep(1) 
        
        print(f"Failed after {MAX_RETRIES} retries.")
        return content 

async def process_single_qa(q, concept_name, llm_description, semaphore):
    prompt_refine = PROMPTS["refine_question"].format(
        CONCEPT=concept_name,
        DESCRIPTION=llm_description,
        QUESTION=q
    )
    
    raw_resp = await query_async(prompt_refine, semaphore)
    refine_question = extract_xml(raw_resp)


    prompt_generate_answer = PROMPTS["generate_answer"].format(
        CONCEPT=concept_name,
        DESCRIPTION=llm_description,
        QUESTION=refine_question.get('refined_question', '')
    )
    
    raw_resp = await query_async(prompt_generate_answer, semaphore)
    answers = extract_xml(raw_resp)

    return {
        "original_question": q,
        "rationale":  refine_question.get('rationale', ''),
        "question":  refine_question.get('refined_question', ''),
        "matching": answers.get('positive_answer', ''),
        "not_matching": answers.get('negative_answer', ''),
        "metadata": answers.get('metadata', '')
    }



async def process_concept(item, semaphore, n_qa_pairs):
    concept_name = item["concept"]
    concept_id = item["concept_id"]
    
    print(f"Starting Concept: {concept_name}")

    prompt_question = PROMPTS["generate_question"].format(CONCEPT=concept_name)
    for i in range(MAX_RETRIES):
        try:
            questions_raw = await query_async(prompt_question, semaphore)
            print("========================")
            print(questions_raw )
            questions_json = extract_json(questions_raw)
            break
        except JsonParseError:
            print(f"\033[91mFailed to parse JSON for {concept_name}\033[0m")
        except Exception as e:
            print(f"Error: {e}. Retrying ({i+1}/{MAX_RETRIES})...")
            await asyncio.sleep(1)

    train_set = questions_json.get("train_questions", [])
    test_set = questions_json.get("test_questions", [])
    llm_description = questions_json.get("description", "")
    
    if n_qa_pairs > 0 and len(train_set) > n_qa_pairs:
        train_set = train_set[:n_qa_pairs]
    if n_qa_pairs > 0 and len(test_set) > n_qa_pairs:
        test_set = test_set[:n_qa_pairs]

    train_tasks = [process_single_qa(q, concept_name, llm_description, semaphore) for q in train_set]
    test_tasks = [process_single_qa(q, concept_name, llm_description, semaphore) for q in test_set]

    all_tasks = train_tasks + test_tasks
    total_qa_count = len(all_tasks)
    
    all_results = []
    if total_qa_count > 0:
        all_results = await tqdm_asyncio.gather(
            *all_tasks, 
            desc=f"[{concept_name}] Generating QAs", 
            leave=False 
        )

    train_results_raw = all_results[:len(train_tasks)]
    test_results_raw = all_results[len(train_tasks):]

    processed_train = []
    refine_train_q = []
    for idx, res in enumerate(train_results_raw):
        res["question_id"] = idx
        res["concept_id"] = concept_id
        res["concept"] = concept_name
        res['llm_description'] = llm_description
        processed_train.append(res)
        refine_train_q.append(res["question"])

    processed_test = []
    refine_test_q = []
    for idx, res in enumerate(test_results_raw):
        res["question_id"] = idx
        res["concept_id"] = concept_id
        res["concept"] = concept_name
        res['llm_description'] = llm_description
        processed_test.append(res)
        refine_test_q.append(res["question"])

    concept_record = {
        "concept": concept_name,
        "concept_id": concept_id,
        "llm_description": llm_description,
        "few_shot_examples": questions_json.get("example", []),
        "train_questions": train_set,
        "test_questions": test_set,
        "refine_train_question": refine_train_q,
        "refine_test_question": refine_test_q
    }

    print(f"Finished Concept: {concept_name}")
    return concept_record, {concept_id: processed_train}, {concept_id: processed_test}


async def main(input_path, output_path, n_concepts, n_qa_pairs):
    concepts = read_json(input_path)
    print(f"Loaded {len(concepts)} concepts.")
    
    if len(concepts) > n_concepts:
        concepts = concepts[:n_concepts]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    all_question = []
    train_results = {}
    test_results = {}

    for item in tqdm_asyncio(concepts, desc="Processing Concepts"):
        result = await process_concept(item, semaphore, n_qa_pairs)
        if result:
            c_record, c_train, c_test = result
            all_question.append(c_record)
            train_results.update(c_train)
            test_results.update(c_test)
    
            # write_file(f"{output_path}/all_question_tmp.json", all_question)

        write_file(f"{output_path}/all_question.json", all_question)
        write_file(f"{output_path}/train_results.json", train_results)
        write_file(f"{output_path}/test_results.json", test_results)
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate question dataset.")
    parser.add_argument('-i','--input_path', default= 'data/case.json',      type=str, help="Path to the input file.")
    parser.add_argument('-o','--output_path', default='case_20', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('-n','--n_concepts', type=int, default=15, help="Number of concepts to generate.")
    parser.add_argument('-a','--n_qa_pairs', type=int, default=100, help="Number of question-answer pairs to generate.")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = input_path.rsplit('/', 1)[0] + f'/result/{args.output_path}'
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Number of concepts: {args.n_concepts}")
    print(f"Number of QA pairs per concept: {args.n_qa_pairs}")
    
    asyncio.run(main(input_path, output_path, n_concepts=args.n_concepts, n_qa_pairs=args.n_qa_pairs))
    
    # python generate_qa_asy.py -i data/version1/language_features/concepts_all.json    -o try -n 2 -a 2
    # python generate_qa_asy.py -i data/version1/personality/concepts_all.json          -o try -n 1 -a 2
    # python generate_qa_asy.py -i data/version1/reasoning_patterns/concepts_all.json   -o try -n 1 -a 2
    # python generate_qa_asy.py -i data/version1/sentiment/concepts_all.json            -o try -n 1 -a 2
    # python generate_qa_asy.py -i data/version1/safety/concepts_all.json               -o try -n 1 -a 2