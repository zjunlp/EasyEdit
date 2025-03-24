from ..utils import build_model_input



def load_contrastive_data(train_data, subset, tokenizer, system_prompt = None, use_chat_template = False):
    dataset = []
    pos_data = [{"text": item["matching"], "label": 1} for item in train_data]
    neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]
    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]
    dataset =  pos_data + neg_data
    for _datum in dataset:
        if type(_datum['text']) != str:
            _datum['text'] = str(_datum['text'])      
        _datum['text'] = build_model_input(_datum['text'], tokenizer, system_prompt, use_chat_template)
            
    
    return dataset


def load_labed_data(train_data, tokenizer, system_prompt = None, use_chat_template = False):
    labels = [int(item['label']) for item in train_data]
    min_label = min(labels)
    max_label = max(labels)
    dataset = []
    for item in train_data:
        mapped_label = (int(item['label']) - min_label) / (max_label - min_label) * 2 - 1
        if type(item['text']) != str:
            item['text'] = str(item['text'])  
        item['text'] = build_model_input(item['text'], tokenizer, system_prompt, use_chat_template)
        dataset.append({'text': item['text'], 'label': mapped_label})
    return dataset


def load_lm_steer_dataset(raw_data, subset, tokenizer, system_prompt, use_chat_template):
    
    if 'text' in raw_data[0] and 'label' in raw_data[0]:
        dataset = load_labed_data(raw_data, tokenizer, system_prompt, use_chat_template)
    elif 'matching' in raw_data[0] and 'not_matching' in raw_data[0]:
        dataset = load_contrastive_data(raw_data, subset, tokenizer, system_prompt, use_chat_template)
    else:
        raise NotImplementedError()

    return dataset
