import os
from numpy import *
import json
import os.path
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import argparse



torch.cuda.set_device(3)

def rewrite_json(path, data):
    with open(path, 'a') as file:
        json.dump(data, file, indent=4)
        file.write(',')
        file.write('\n')
        file.flush()


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data      


def write_json(path, data, case_id = None, data_all = None):
    if data_all is None:
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(path, 'a') as file:
            if case_id == 0:
                file.write("[")
            json.dump(data, file, indent=4)
            if case_id == data_all-1:
                file.write('\n')
                file.write("]")
            else:
                file.write(',')
                file.write('\n')
                file.flush()

def load_model(model_ckpt = None, 
               tokenizer_ckpt = None):
    """
    Load model, tokenizer.
    """
    model = LlamaForCausalLM.from_pretrained(model_ckpt, output_hidden_states=True).to('cuda')
    tok = LlamaTokenizer.from_pretrained(tokenizer_ckpt)
    tok.pad_token_id = tok.eos_token_id
        
    return model, tok



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_LLM_ckpt', required=True, type=str) 
    parser.add_argument('--tok_ckpt', required=True, type=str) 
    parser.add_argument('--suffix_system_prompt', default=None, type=str) 
    parser.add_argument('--data_dir', default='./data/SafeEdit_test_ALL.json', type=str)
    parser.add_argument("--max_output_length", type=int, default=600) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--results_save_dir', required=True, type=str)
    
    args = parser.parse_args()


    output_dir = f'{args.results_save_dir}/detoxify_val.json'
    if not os.path.exists(args.results_save_dir):
        os.mkdir(args.results_save_dir)
    print(f"Results will be stored at {args.results_save_dir}")

    model, tokenizer = load_model(args.edited_LLM_ckpt, args.tok_ckpt)
    tokenizer.padding_side = 'left'
    model.eval()
    data = read_json(args.data_dir)
    data = data[0:1]
    for i in range(0, len(data), args.batch_size):
        if i + args.batch_size > len(data):
            batch = data[i:len(data)]
        else:
            batch = data[i:i + args.batch_size]
        if args.suffix_system_prompt is not None:
            test_prompt = [item["malicious input"] + args.suffix_system_prompt for item in batch]
        else:
            test_prompt = [item["malicious input"] for item in batch]
        input = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = model.generate(**input, max_new_tokens=args.max_output_length)
            texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
        for index, res in enumerate(only_response):
            batch[index]["response"] = res
            write_json(output_dir, batch[index], i+index, len(data))
    print(f'test is all done')
    

# example
# python test_detoxify_generate.py --edited_LLM_ckpt ./safety_results/dinm_llama2-chat --tok_ckpt ./hugging_cache/llama-2-7b --results_save_dir ./safety_results


    

            
            






