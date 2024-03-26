from easyeditor import (
    PEREditTrainer, 
    MENDTrainingHparams, 
    MENDHyperParams,
    PersonalityDataset, 
    IKEHyperParams,
    PerEditor
)

import os, json, sys
import argparse
import numpy as np
import openai
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

os.environ["OPENAI_API_KEY"] = "sk-xxx" # setting your own apikey for pae evaluation


def train_run_MEND(data_dir, hparams_path):
    
    training_hparams = MENDTrainingHparams.from_hparams(hparams_path)
    train_ds = PersonalityDataset(os.path.join(data_dir, './PersonalityEdit/train.json'), config=training_hparams)
    eval_ds = PersonalityDataset(os.path.join(data_dir, './PersonalityEdit/val.json'), config=training_hparams)
    trainer = PEREditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
        
    
def test_run_MEND(data_dir, hparams_path):
    hparams = MENDHyperParams.from_hparams(hparams_path)
    eval_ds = PersonalityDataset(os.path.join(data_dir, './PersonalityEdit/test.json'), config=hparams)
    editor = PerEditor.from_hparams(hparams)
    metrics, _, _ = editor.edit_dataset(
        ds=eval_ds,
        keep_original_weight=True        
    )
    return metrics

    
    
def edit_IKE(data_dir, hparams_path):
    hparams = IKEHyperParams.from_hparams(hparams_path)
    eval_ds = PersonalityDataset(os.path.join(data_dir, './PersonalityEdit/test.json'), config=hparams)
    editor = PerEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        keep_original_weight=True        
    )
    return metrics


def PAE(pre_text, edit_text, target_per, retry=4, coherent=True):
    
    if not coherent: 
        return {
        "pre_gpt_score": None,
        "edit_gpt_score": None,
        "pae": None
    }

    def call_gpt4(text):
        while True:
            try:
                response=openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                            {"role": "user", "content": text }
                        ], 
                    temperature=0,
                    max_tokens=1 
                    )
                break
            except Exception as e:
                print(e)

        score = response["choices"][0]["message"]["content"]
        cost = response["usage"]["total_tokens"]
        score = int(score) if score in ["1", "2", "3", "4", "5"] else -1
        
        return score, cost
    
    prompt_dict = {
        "extraversion": "Respond to match this description.\nPersona Description: I can be described as a person with one or more adjectives in friendly, extraverted, talkative, bold, assertive, active, energetic, adventurous and daring, cheerful.\nEvaluating the opinion: \"{}\"\nhow accurately the opinion mathes the description, please rate a scale in [1,2,3,4,5] (where 1 = `very inaccurate`, 2 = `moderately inaccurate`, 3 = `neither accurate nor inaccurate`, 4 = `moderately accurate`, and 5 = `very accurate`):",
        "agreeableness": "Respond to match this description.\nPersona Description: I can be described as a person with one or more adjectives in trustful, dishonest, honest, altruistic, generous, cooperative, humble, sympathetic, unselfish, agreeable.\nEvaluating the opinion: \"{}\"\nhow accurately the opinion mathes the description, please rate a scale in [1,2,3,4,5] (where 1 = `very inaccurate`, 2 = `moderately inaccurate`, 3 = `neither accurate nor inaccurate`, 4 = `moderately accurate`, and 5 = `very accurate`):",
        "neuroticism": "Respond to match this description.\nPersona Description: I can be described as a person with one or more adjectives in tense, nervous, anxious, angry, irritable, depressed, self-conscious, impulsive, discontented, emotionally unstable.\nEvaluating the opinion: \"{}\"\nhow accurately the opinion mathes the description, please rate a scale in [1,2,3,4,5] (where 1 = `very inaccurate`, 2 = `moderately inaccurate`, 3 = `neither accurate nor inaccurate`, 4 = `moderately accurate`, and 5 = `very accurate`):"
    }
    
    prompt = prompt_dict[target_per]
    for i in range(retry):
        pre_score, _ = call_gpt4(prompt.format(pre_text))
        if pre_score != -1: break
        pre_score = None
                
    for i in range(retry):
        edit_score, _ = call_gpt4(prompt.format(edit_text))
        if edit_score != -1: break
        edit_score = None
            
    result = {
        "pre_gpt_score": pre_score,
        "edit_gpt_score": edit_score,
        "pae": edit_score-pre_score
    }
    
    return result
    
    
def TPEI(model, tokenizer, pre_text, edit_text, target_per, coherent=True):
    
    if not coherent: return {"acc": None, "tpei":None}
    
    device = model.device
    label_to_id = {"neuroticism":0, "agreeableness":1, "extraversion":2}
    pre_text_input = tokenizer(pre_text, padding="max_length", max_length=128, truncation=True, return_tensors="pt",).to(device)
    edit_text_input = tokenizer(edit_text, padding="max_length", max_length=128, truncation=True, return_tensors="pt",).to(device)
    label = label_to_id[target_per]
    pre_text_input["labels"] = torch.tensor([label], dtype=torch.long).to(device)
    edit_text_input["labels"] = torch.tensor([label], dtype=torch.long).to(device)
    
    with torch.no_grad():
        pre_output = model(**pre_text_input)
        edit_output = model(**edit_text_input)
        prediction = torch.argmax(torch.nn.functional.softmax(edit_output.logits, dim=-1)).item()
        tpsi = pre_output.loss.item() - edit_output.loss.item()
        acc = int(label==prediction)
    
    return {
        "acc": acc,
        "tpei": tpsi
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_path', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--metric_file', default=None, type=str)
    parser.add_argument('--TPEI', action="store_true")
    parser.add_argument('--PAE', action="store_true")
    parser.add_argument('--cls_path', default=None, type=str)
    args = parser.parse_args()


    if args.editing_method == 'IKE':
        edit_func = edit_IKE
    elif args.editing_method == 'MEND_train':
        edit_func = train_run_MEND
    elif args.editing_method == 'MEND_test':
        edit_func = test_run_MEND
    else:
        raise NotImplementedError
    
    
    if not args.metric_file:
        metrics = edit_func(args.data_dir, args.hparams_path)
        if "train" not in args.editing_method:
            json.dump(metrics, open(f"./{args.editing_method}_metrics.json", "w"), ensure_ascii=False, indent=4)
    else:
        metrics = json.load(open(args.metric_file))
    
    if args.TPEI:
        assert args.cls_path is not None
        model = AutoModelForSequenceClassification.from_pretrained(args.cls_path).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
        model.eval()
        for metric in metrics:
            metric.update(TPEI(
                model=model,
                tokenizer=tokenizer,
                pre_text=metric["pre_text"],
                edit_text=metric["edit_text"],
                target_per=metric["target_per"],
                coherent=metric["coherent"] # skip the incoherent case
            ))
    
    if args.PAE:
        for metric in metrics:
            metric.update(PAE(
                pre_text=metric["pre_text"],
                edit_text=metric["edit_text"],
                target_per=metric["target_per"],
                coherent=metric["coherent"] # skip the incoherent case
            ))
    
    for met in ["es", "dd", "acc", "tpei", "pae"]:
        if met not in metrics[0].keys(): continue
        mets = [metric[met] for metric in metrics if metric[met] is not None] 
        print(f"{met}:{np.mean(mets)}") 
