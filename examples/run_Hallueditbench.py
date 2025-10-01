import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_dtype(torch.float32)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

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

SYSTEM_MSG_QA = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."
SYSTEM_MSG_EVAL = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically; otherwise, output '0'. Do not repeat the question or provide any explanation."
SYSTEM_MSG_MULTIPLE_CHOICE = "Always respond to the multiple-choice question by selecting from the provided options. Only output the choice letter (A, B, C, or D)."

def get_response(hparams, model, tok, messages, max_new_tokens=16, eval_flag=False, device_eval='cuda:0'): 
    if eval_flag:
        device = device_eval
    else:
        if hasattr(hparams, 'device'):
            if isinstance(hparams.device, int):
                device = f"cuda:{hparams.device}"
            elif isinstance(hparams.device, str) and hparams.device.isdigit():
                device = f"cuda:{hparams.device}"
            else:
                device = hparams.device
        else:
            device = "cuda:0"  
    
    terminators = [tok.eos_token_id]
    if tok.convert_tokens_to_ids("<|eot_id|>") is not None:
        terminators.append(tok.convert_tokens_to_ids("<|eot_id|>"))
    
    try:
        if 'gpt' in hparams.model_name.lower() and eval_flag is False:
            msg_tokenized = tok(messages[0], return_tensors='pt').to(device)
        else:
            msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
        
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            msg_tokenized = {k: v.to(dtype=torch.float32) if v.dtype in [torch.bfloat16, torch.float16] else v 
                            for k, v in msg_tokenized.items()}
        
        with torch.no_grad():
            output_ids = model.generate(
                **msg_tokenized, 
                max_new_tokens=max_new_tokens, 
                eos_token_id=terminators, 
                do_sample=False, 
                pad_token_id=tok.eos_token_id,
                use_cache=True
            )
        
        return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')
    
    except Exception as e:
        print(f"Error in get_response: {e}")
        raise e

def evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_qa, label, device_eval):
    """Enhanced evaluation function with stricter matching logic"""
    if 'gpt' in hparams.model_name.lower():
        for substr in ["Question:", "Do not"]:
            if substr in output_qa:
                output_qa = output_qa[:output_qa.find(substr)]

    if output_qa.strip().lower() == label.strip().lower():
        response_eval = 1
        match_type = "Exact Match"
    # Try partial matching with stricter conditions
    elif (len(label) > 2 and label.lower() in output_qa.lower()) or \
         (len(output_qa) > 2 and output_qa.lower() in label.lower()):
        # Avoid short string mismatches by adding length checks
        response_eval = 1
        match_type = "Partial Match"
    else:
        # Semantic matching as last resort
        user_msg_eval = f"""Text 1: {label} \nText 2: {output_qa}"""
        messages_eval = [{"role": "system", "content": SYSTEM_MSG_EVAL}, {"role": "user", "content": user_msg_eval}]
        response_eval = get_response(hparams, model_eval, tok_eval, messages_eval, max_new_tokens=1, eval_flag=True, device_eval=device_eval)
        
        if str(response_eval) == '1':
            match_type = "Semantic Match"
        else:
            match_type = "No Match"
            response_eval = 0

    print(f"===== Question: {prompt_qa} | Prediction: {output_qa} | Label: {label} | Evaluation: {response_eval} ({match_type}) =====")
    
    if str(response_eval) not in ['0', '1']:
        response_eval = 0
    return int(response_eval), output_qa

def test_prediction_acc(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, system_msg=SYSTEM_MSG_QA):
    model_qa_name = hparams.model_name
    user_msg_qa = prompt_qa
    
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg_qa}]
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": system_msg + ' ' + user_msg_qa}]
    elif 'vicuna' in model_qa_name.lower() or 'gpt' in model_qa_name.lower():
        messages_qa = [f"{system_msg} Question: {user_msg_qa} Answer:"]
    else:
        messages_qa = [system_msg + ' ' + user_msg_qa]

    output_qa = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)

    if label is None:  
        return None, output_qa
    
    return evaluate_response(hparams, model_eval, tok_eval, prompt_qa, output_qa, label, device_eval)

def test_prediction_acc_multi_turn(hparams, model_qa, tok_qa, model_eval, tok_eval, device_eval, prompt_qa, label, multi_turn, yes_question):
    """Robustness Test"""
    acc_ls = []
    output_qa_ls = []
    
    if multi_turn == 'sure':
        questions_follow_up = [f"Are you sure about your answer to the original question? Respond with 'Yes' or 'No.'" for _ in range(10)]
    elif multi_turn == 'yes':
        questions_follow_up = [f'Your answer to the original question is wrong. {yes_question} Respond with "Yes" or "No."' for _ in range(10)] 
    
    label_follow_up = 'Yes'

    model_qa_name = hparams.model_name
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": SYSTEM_MSG_QA}, {"role": "user", "content": prompt_qa}]
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": SYSTEM_MSG_QA + ' ' + prompt_qa}]
    else:
        messages_qa = [SYSTEM_MSG_QA + ' ' + prompt_qa]

    # First Answer
    current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)
    eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, prompt_qa, current_output, label, device_eval)
    acc_ls.append(eval_acc)
    output_qa_ls.append(current_output)

    # Follow-Up Answers
    for question in questions_follow_up:
        messages_qa.append({"role": "assistant", "content": current_output})
        messages_qa.append({"role": "user", "content": question})
        
        current_output = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)
        eval_acc, _ = evaluate_response(hparams, model_eval, tok_eval, question, current_output, label_follow_up, device_eval)
        acc_ls.append(eval_acc)
        output_qa_ls.append(current_output)
        
    return acc_ls, output_qa_ls

def compute_locality_score(pre_output, post_output, hparams, model_eval, tok_eval, device_eval, question):
    """Calculate locality Score - whether the model's output is consistent with the original output"""
    acc, _ = evaluate_response(hparams, model_eval, tok_eval, question, pre_output, post_output, device_eval)
    return acc

def compute_edit_quality_standard(
    hparams, model, tok, model_eval, tok_eval, device_eval, record, multi_turn=None, pre_or_post='pre'
):
    ret = {}
    
    # 1. EFFICACY
    edit_prompt = record["prompt"]
    target_new = record["target_new"]
    
    if multi_turn is not None and pre_or_post == 'post':
        yes_question = record.get('yes_question')
        acc_ls, output_ls = test_prediction_acc_multi_turn(
            hparams, model, tok, model_eval, tok_eval, device_eval, 
            edit_prompt, target_new, multi_turn, yes_question
        )
        ret['edit_acc'] = [acc_ls[0]]  
        ret['edit_output'] = [output_ls[0]]
        ret['edit_acc_multi_turn'] = acc_ls  
        ret['edit_output_multi_turn'] = output_ls
    else:
        # Single Turn Test
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, edit_prompt, target_new)
        ret['edit_acc'] = [acc]
        ret['edit_output'] = [output]
    
    # Initialize all categories
    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_questions'] = {}
    ret['no_questions'] = {}
    ret['multiple_choice_questions'] = {}
    ret['reversed_relation_questions'] = {}
    ret['questions_2hop'] = {}
    ret['questions_3hop'] = {}
    ret['questions_4hop'] = {}
    ret['questions_5hop'] = {}
    ret['questions_6hop'] = {}
    
    if 'rephrase_prompt' in record and record['rephrase_prompt'] is not None:
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, record['rephrase_prompt'], target_new)
        ret['rephrase_acc'] = [acc]
        ret['rephrase_output'] = [output]
    
    if 'yes_question' in record and record['yes_question'] is not None:
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, record['yes_question'], 'Yes')
        ret['yes_questions']['yes_acc'] = [acc]
        ret['yes_questions']['yes_output'] = [output]
    
    if 'no_question' in record and record['no_question'] is not None:
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, record['no_question'], 'No')
        ret['no_questions']['no_acc'] = [acc]
        ret['no_questions']['no_output'] = [output]
    
    # Fix: Multiple choice questions need to include the original question
    if 'multiple_choice_question' in record and record['multiple_choice_question'] is not None:
        # Build complete multiple choice format: original question + options
        base_question = record.get('base_question', record['prompt'])  # Get original question
        full_mcq = f"{base_question} {record['multiple_choice_question']}"
        
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, 
                                         full_mcq, record.get('multiple_choice_label'), SYSTEM_MSG_MULTIPLE_CHOICE)
        ret['multiple_choice_questions']['multiple_choice_acc'] = [acc]
        ret['multiple_choice_questions']['multiple_choice_output'] = [output]
    
    if 'reversed_relation_question' in record and record['reversed_relation_question'] is not None:
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, 
                                         record['reversed_relation_question'], record['subject'])
        ret['reversed_relation_questions']['reversed_relation_acc'] = [acc]
        ret['reversed_relation_questions']['reversed_relation_output'] = [output]
    
    for hop in ['2hop', '3hop', '4hop', '5hop', '6hop']:
        question_key = f'question_{hop}'
        answer_key = f'answer_{hop}'
        if question_key in record and record[question_key] is not None:
            acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, 
                                             record[question_key], record.get(answer_key))
            ret[f'questions_{hop}'][f'{hop}_acc'] = [acc]
            ret[f'questions_{hop}'][f'{hop}_output'] = [output]
    
    if 'locality_question' in record and record['locality_question'] is not None:
        acc, output = test_prediction_acc(hparams, model, tok, model_eval, tok_eval, device_eval, 
                                         record['locality_question'], None)
        ret['locality']['locality_acc'] = [acc] if acc is not None else [0]
        ret['locality']['locality_output'] = [output]
    
    return ret

def prepare_edit_requests_standard(df):
    """Fix data preparation by adding base_question field"""
    requests = []
    
    for _, row in df.iterrows():
        relation = row['relation']
        subject = row['subject']
        object_val = row['object']
        
        # Construct the main edit prompt
        if 'industry' in relation.lower():
            prompt = f"What is the industry of {subject}?"
        elif 'headquarters' in relation.lower() or 'location' in relation.lower():
            prompt = f"Where is the headquarters of {subject}?"
        elif 'founded' in relation.lower():
            prompt = f"When was {subject} founded?"
        elif 'founder' in relation.lower():
            prompt = f"Who founded {subject}?"
        elif 'birth' in relation.lower() or 'born' in relation.lower():
            prompt = f"Where was {subject} born?"
        else:
            prompt = f"What is the {relation} of {subject}?"
        
        request = {
            'subject': subject,
            'prompt': prompt,
            'target_new': object_val,
            'ground_truth': object_val,
            
            # Add base_question for multiple choice questions
            'base_question': row.get('question', prompt),  # Original question
            
            # GENERALIZATION 
            'rephrase_prompt': row.get('paraphrased_question'),
            'yes_question': row.get('yes_question'),
            'no_question': row.get('no_question'),
            'multiple_choice_question': row.get('multiple_choice_with_letters'),
            'multiple_choice_label': row.get('multiple_choice_labels'),
            'reversed_relation_question': row.get('reversed_relation_question'),
            
            # PORTABILITY 
            'question_2hop': row.get('question_2hop'),
            'answer_2hop': row.get('answer_2hop'),
            'question_3hop': row.get('question_3hop'),
            'answer_3hop': row.get('answer_3hop'),
            'question_4hop': row.get('question_4hop'),
            'answer_4hop': row.get('answer_4hop'),
            'question_5hop': row.get('question_5hop'),
            'answer_5hop': row.get('answer_5hop'),
            'question_6hop': row.get('question_6hop'),
            'answer_6hop': row.get('answer_6hop'),
            
            # LOCALITY Data
            'locality_question': row.get('locality_question')
        }
        
        requests.append(request)
    
    return requests

def calculate_summary_metrics_standard(all_metrics):
    """Calculate summary metrics for standard evaluation"""
    if not all_metrics:
        return {}
    
    summary = {}
    
    for phase in ['pre', 'post']:
        summary[phase] = {}
        
        # EFFICACY 
        if 'edit_acc' in all_metrics[0][phase]:
            edit_accs = [m[phase]['edit_acc'][0] for m in all_metrics if 'edit_acc' in m[phase]]
            summary[phase]['edit_acc'] = np.mean(edit_accs) if edit_accs else 0.0
        
        # GENERALIZATION - Rephrase 
        if 'rephrase_acc' in all_metrics[0][phase]:
            rephrase_accs = [m[phase]['rephrase_acc'][0] for m in all_metrics if 'rephrase_acc' in m[phase]]
            summary[phase]['rephrase_acc'] = np.mean(rephrase_accs) if rephrase_accs else 0.0
        
        # ROBUSTNESS
        if phase == 'post' and 'edit_acc_multi_turn' in all_metrics[0][phase]:
            max_turns = max(len(m[phase]['edit_acc_multi_turn']) for m in all_metrics if 'edit_acc_multi_turn' in m[phase])
            turn_accuracies = []
            for turn_idx in range(max_turns):
                turn_accs = [m[phase]['edit_acc_multi_turn'][turn_idx] for m in all_metrics 
                           if 'edit_acc_multi_turn' in m[phase] and turn_idx < len(m[phase]['edit_acc_multi_turn'])]
                if turn_accs:
                    turn_accuracies.append(np.mean(turn_accs))
            summary[phase]['edit_acc_multi_turn'] = turn_accuracies
        
        # LOCALITY 
        summary[phase]['locality'] = {}
        if 'locality' in all_metrics[0][phase]:
            loc_accs = [m[phase]['locality']['locality_acc'][0] for m in all_metrics 
                       if 'locality' in m[phase] and 'locality_acc' in m[phase]['locality']]
            if loc_accs:
                summary[phase]['locality']['locality_acc'] = np.mean(loc_accs)
        
        # GENERALIZATION - Yes/No 
        for question_type in ['yes_questions', 'no_questions']:
            summary[phase][question_type] = {}
            if question_type in all_metrics[0][phase]:
                for key in all_metrics[0][phase][question_type]:
                    if key.endswith('_acc'):
                        accs = [m[phase][question_type][key][0] for m in all_metrics 
                               if question_type in m[phase] and key in m[phase][question_type]]
                        if accs:
                            summary[phase][question_type][key] = np.mean(accs)
        
        # GENERALIZATION - Multiple Choice
        summary[phase]['multiple_choice_questions'] = {}
        if 'multiple_choice_questions' in all_metrics[0][phase]:
            for key in all_metrics[0][phase]['multiple_choice_questions']:
                if key.endswith('_acc'):
                    accs = [m[phase]['multiple_choice_questions'][key][0] for m in all_metrics 
                           if 'multiple_choice_questions' in m[phase] and key in m[phase]['multiple_choice_questions']]
                    if accs:
                        summary[phase]['multiple_choice_questions'][key] = np.mean(accs)
        
        # GENERALIZATION - Reversed Relation
        summary[phase]['reversed_relation_questions'] = {}
        if 'reversed_relation_questions' in all_metrics[0][phase]:
            for key in all_metrics[0][phase]['reversed_relation_questions']:
                if key.endswith('_acc'):
                    accs = [m[phase]['reversed_relation_questions'][key][0] for m in all_metrics 
                           if 'reversed_relation_questions' in m[phase] and key in m[phase]['reversed_relation_questions']]
                    if accs:
                        summary[phase]['reversed_relation_questions'][key] = np.mean(accs)
        
        # PORTABILITY - Multi-hop Reasoning
        for hop_type in ['questions_2hop', 'questions_3hop', 'questions_4hop', 'questions_5hop', 'questions_6hop']:
            summary[phase][hop_type] = {}
            if hop_type in all_metrics[0][phase]:
                for key in all_metrics[0][phase][hop_type]:
                    if key.endswith('_acc'):
                        accs = [m[phase][hop_type][key][0] for m in all_metrics 
                               if hop_type in m[phase] and key in m[phase][hop_type]]
                        if accs:
                            summary[phase][hop_type][key] = np.mean(accs)
    
    return summary

def halluedit_standard_edit_and_evaluate(
    editing_method: str,
    hparams_dir: str,
    data_path: str,
    eval_model_id: str = '/models/llama3-8B-Instruct',
    data_size: int = None,
    results_dir: str = './results',
    multi_turn: str = 'sure' 
):
    print(f"Starting {editing_method} editing experiment with HalluEditBench standard...")
    start_time = time.time()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 1. Load editing method
    if editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif editing_method == 'SERAC':
        editing_hparams = SERACHparams
    else:
        raise NotImplementedError(f"Method {editing_method} not implemented")
    
    hparams = editing_hparams.from_hparams(hparams_dir)
    
    if hasattr(hparams, 'device') and isinstance(hparams.device, int):
        hparams.device = f"cuda:{hparams.device}"
    elif not hasattr(hparams, 'device'):
        hparams.device = "cuda:0"  
    
    # 2. Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    if data_size is not None:
        df = df[:data_size]
        
    print(f"Loaded {len(df)} records")
    
    # 3. Prepare edit requests
    requests = prepare_edit_requests_standard(df)
    
    # 4. Load model
    print("Loading editor...")
    editor = BaseEditor.from_hparams(hparams)
    
    print("Loading evaluation model...")
    model_eval = AutoModelForCausalLM.from_pretrained(
        eval_model_id, 
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    
    tok_eval = AutoTokenizer.from_pretrained(eval_model_id, trust_remote_code=True)
    if tok_eval.pad_token is None:
        tok_eval.pad_token = tok_eval.eos_token
    
    if model_eval.dtype != torch.float32:
        model_eval = model_eval.to(dtype=torch.float32)
    
    device_eval = "cuda:0"
    print(f"Using evaluation device: {device_eval}")
    
    # 5. Pre-edit evaluation
    print("\n" + "="*60)
    print("PHASE 1: PRE-EDIT EVALUATION")
    print("="*60)
    
    all_metrics = []
    pre_locality_outputs = []  
    
    for i, request in enumerate(requests):
        print(f"\nEvaluating request {i+1}/{len(requests)}: {request['subject']} -> {request['target_new']}")
        pre_metrics = compute_edit_quality_standard(hparams, editor.model, editor.tok, model_eval, tok_eval, device_eval, request, pre_or_post='pre')
        
        if 'locality' in pre_metrics and 'locality_output' in pre_metrics['locality']:
            pre_locality_outputs.append(pre_metrics['locality']['locality_output'][0])
        else:
            pre_locality_outputs.append(None)
        
        metrics = {
            'case_id': i,
            'requested_edit': request,
            'pre': pre_metrics
        }
        all_metrics.append(metrics)
    
    # 6. Edit
    print("\n" + "="*60)
    print("PHASE 2: KNOWLEDGE EDITING")
    print("="*60)
    
    for i, request in enumerate(requests):
        print(f"\nEditing {i+1}/{len(requests)}: {request['prompt']} -> {request['target_new']}")
        
        edit_start = time.time()
        
        # Edit
        if editing_method in ['IKE', 'ICL']:
            edited_model, weights_copy = editor.model, {}
        else:
            edited_model, weights_copy = editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=True
            )
        
        edit_time = time.time() - edit_start
        print(f"Edit completed in {edit_time:.2f}s")
        
        # 7. Post-edit evaluation
        print(f"Evaluating edited model with multi-turn robustness testing...")
        post_metrics = compute_edit_quality_standard(hparams, edited_model, editor.tok, model_eval, tok_eval, device_eval, request, multi_turn=multi_turn, pre_or_post='post')
        
        # 8. Compute locality score
        if ('locality' in post_metrics and 'locality_output' in post_metrics['locality'] and 
            pre_locality_outputs[i] is not None):
            
            pre_output = pre_locality_outputs[i]
            post_output = post_metrics['locality']['locality_output'][0]
            locality_score = compute_locality_score(
                pre_output, post_output, hparams, model_eval, tok_eval, device_eval, 
                request['locality_question']
            )
            post_metrics['locality']['locality_acc'] = [locality_score]
            print(f"Locality: Pre='{pre_output}' | Post='{post_output}' | Score={locality_score}")
        
        all_metrics[i].update({
            'post': post_metrics,
            'edit_time': edit_time
        })
        
        # Restore original weights
        if editing_method == 'KN' or (editing_method == 'GRACE'):
            with torch.no_grad():
                weights_copy()
        elif editing_method == 'LoRA':
            edited_model.unload()
            if hasattr(editor.model, 'peft_config'):
                del editor.model.peft_config
        elif editing_method != 'IKE':
            with torch.no_grad():
                for k, v in weights_copy.items():
                    editor.model.get_parameter(k)[...] = v.to(f"cuda:{hparams.device}")
    
    # 9. Compute summary metrics
    print("\n" + "="*60)
    print("PHASE 3: SUMMARY CALCULATION")
    print("="*60)
    
    summary_metrics = calculate_summary_metrics_standard(all_metrics)
    
    # 10. Print results
    print("\n" + "="*60)
    print("FINAL RESULTS (HalluEditBench Standard)")
    print("="*60)
    
    for phase in ['pre', 'post']:
        print(f"\n{phase.upper()} EDIT:")
        metrics = summary_metrics[phase]
        
        # EFFICACY
        if 'edit_acc' in metrics:
            print(f"  EFFICACY (edit_acc): {metrics['edit_acc']:.3f}")
        if 'rephrase_acc' in metrics:
            print(f"  GENERALIZATION (rephrase_acc): {metrics['rephrase_acc']:.3f}")
        
        # ROBUSTNESS - Multi-turn testing (only post-edit)
        if phase == 'post' and 'edit_acc_multi_turn' in metrics:
            print(f"  ROBUSTNESS (Multi-turn):")
            for turn_idx, acc in enumerate(metrics['edit_acc_multi_turn']):
                print(f"    Turn {turn_idx+1}: {acc:.3f}")
            # Compute average robustness
            avg_robustness = np.mean(metrics['edit_acc_multi_turn'][1:]) 
            print(f"    Average (turns 2-11): {avg_robustness:.3f}")
        
        # LOCALITY
        if 'locality' in metrics and 'locality_acc' in metrics['locality']:
            print(f"  LOCALITY: {metrics['locality']['locality_acc']:.3f}")
        
        # GENERALIZATION - Yes/No 
        for question_type in ['yes_questions', 'no_questions']:
            if question_type in metrics and metrics[question_type]:
                print(f"  {question_type.replace('_', ' ').title()}:")
                for key, val in metrics[question_type].items():
                    print(f"    {key}: {val:.3f}")
        
        # GENERALIZATION - Multiple Choice and Reversed Relation
        if 'multiple_choice_questions' in metrics and metrics['multiple_choice_questions']:
            print(f"  Multiple Choice Questions:")
            for key, val in metrics['multiple_choice_questions'].items():
                print(f"    {key}: {val:.3f}")
        
        if 'reversed_relation_questions' in metrics and metrics['reversed_relation_questions']:
            print(f"  Reversed Relation Questions:")
            for key, val in metrics['reversed_relation_questions'].items():
                print(f"    {key}: {val:.3f}")
        
        # PORTABILITY - Multi-hop Reasoning
        print(f"  PORTABILITY (Multi-hop Reasoning):")
        for hop_type in ['questions_2hop', 'questions_3hop', 'questions_4hop', 'questions_5hop', 'questions_6hop']:
            if hop_type in metrics and metrics[hop_type]:
                for key, val in metrics[hop_type].items():
                    hop_num = hop_type.replace('questions_', '').replace('hop', '')
                    print(f"    {hop_num}-hop {key}: {val:.3f}")
    
    # 11. Save results
    os.makedirs(results_dir, exist_ok=True)
    model_name = hparams.model_name.split('/')[-1].replace('-', '_').lower()
    topic_name = os.path.basename(data_path).replace('.csv', '')
    result_file = f"{results_dir}/{model_name}_{topic_name}_{editing_method}_{multi_turn}_fixed.json"
    
    final_results = {
        'summary_metrics': summary_metrics,
        'detailed_metrics': all_metrics,
        'config': {
            'editing_method': editing_method,
            'model_name': hparams.model_name,
            'data_path': data_path,
            'data_size': len(df),
            'multi_turn': multi_turn,
            'total_time': time.time() - start_time
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {result_file}")
    print(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
    
    del model_eval
    del editor
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HalluEditBench Standard Knowledge Editing Evaluation (Fixed)')
    parser.add_argument('--editing_method', required=True, type=str, 
                       choices=['FT', 'IKE', 'KN', 'MEMIT', 'ROME', 'LoRA', 'MEND', 'SERAC'],
                       help='Knowledge editing method to use')
    parser.add_argument('--hparams_dir', required=True, type=str,
                       help='Directory containing hyperparameter files')
    parser.add_argument('--data_path', required=True, type=str,
                       help='Path to CSV data file')
    parser.add_argument('--eval_model_id', default='/disk1/xuhaoming/models/llama3-8B-Instruct', type=str,
                       help='Model ID for evaluation')
    parser.add_argument('--data_size', default=None, type=int,
                       help='Limit number of examples to process')
    parser.add_argument('--results_dir', default='./results', type=str,
                       help='Directory to save results')
    parser.add_argument('--multi_turn', default='sure', choices=['sure', 'yes'], type=str,
                       help='Multi-turn robustness test type: sure=confidence, yes=correction')
    
    args = parser.parse_args()
    
    results = halluedit_standard_edit_and_evaluate(
        editing_method=args.editing_method,
        hparams_dir=args.hparams_dir,
        data_path=args.data_path,
        eval_model_id=args.eval_model_id,
        data_size=args.data_size,
        results_dir=args.results_dir,
        multi_turn=args.multi_turn
    )
