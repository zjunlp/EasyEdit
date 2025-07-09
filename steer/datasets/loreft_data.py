from ..utils import build_model_input


def load_loreft_data(train_data, subset, tokenizer, system_prompt = None, use_chat_template = False):
    dataset = []
    new_dataset = []
    pos_data = [{"input":item["question"],"output": item["matching"], "label": 1} for item in train_data]
    if subset is not None:
        pos_data = pos_data[:subset]
    dataset =  pos_data # loreft just needs pos data
    for _datum in dataset:
        if type(_datum['input']) != str:
            _datum['input'] = str(_datum['input']) 
        if type(_datum['output']) != str:
            _datum['output'] = str(_datum['output'])
        inputs = build_model_input(_datum["input"], tokenizer, system_prompt, use_chat_template, _datum["output"])
        prompts = build_model_input(_datum["input"], tokenizer, system_prompt, use_chat_template)

        new_dataset.append({"input": inputs, "prompt": prompts,"output": _datum["output"], "label": _datum["label"]})
            
    return new_dataset

def load_reft_eval_data(eval_data, subset, tokenizer, system_prompt = None, use_chat_template = False):
    dataset = []
    new_dataset = []
    data = [{"input":item["input"]} for item in eval_data]
    if subset is not None:
        data = data[:subset]
    dataset =  data
    for _datum in dataset:
        if type(_datum['input']) != str:
            _datum['input'] = str(_datum['input'])
        inputs = build_model_input(_datum["input"], tokenizer, system_prompt, use_chat_template)
        new_dataset.append({"input": inputs})
    return new_dataset