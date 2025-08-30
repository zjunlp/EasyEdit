from ..utils import build_model_input, build_multimodal_model_input
from PIL import Image


def load_contrastive_data(train_data, subset, tokenizer, system_prompt = None, use_chat_template = False):
    dataset = []
    pos_data = [{"text": item["matching"], "label": 1} for item in train_data]
    neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]
    if "question" in train_data[0]:
        ques_data = [{"text": item.get("question", "")} for item in train_data]
        ques_data = ques_data + ques_data
    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]
    dataset =  pos_data + neg_data
    for i, _datum in enumerate(dataset):
        if type(_datum['text']) != str:
            _datum['text'] = str(_datum['text'])
        if "question" in train_data[0]:
            _datum['text'] = ques_data[i]['text'] + _datum['text']
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

#Add multimodal support load

def load_multimodal_contrastive_data(train_data, subset, processor, system_prompt = None, use_chat_template = False):
    dataset = []
    pos_data = [{"text": item["matching"], "label": 1} for item in train_data]
    neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]
    
    # 处理图像数据
    if "image" in train_data[0]:
        pos_data = [{"text": item["matching"], "image": item.get("image") if isinstance(item.get("image"),Image.Image) else Image.open(item.get("image")), "label": 1} for item in train_data]
        neg_data = [{"text": item["not_matching"], "image": item.get("image") if isinstance(item.get("image"),Image.Image) else Image.open(item.get("image")), "label": -1} for item in train_data]
    
    if "question" in train_data[0]:
        ques_data = [{"text": item.get("question", "")} for item in train_data]
        ques_data = ques_data + ques_data
    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]
    dataset =  pos_data + neg_data
    for i, _datum in enumerate(dataset):
        if type(_datum['text']) != str:
            _datum['text'] = str(_datum['text'])
        
        # 构建对话消息
        if "question" in train_data[0]:
            if "image" in train_data[0]:
                conversation = [
                    {"role": "user", "content": [{"type": "text", "text": ques_data[i]['text']}, {"type": "image"}]},
                    {"role": "assistant", "content": _datum['text']}
                ]
            else:
                conversation = [
                    {"role": "user", "content": [{"type": "text", "text": ques_data[i]['text']}]},
                    {"role": "assistant", "content": _datum['text']}
                ]
        else:
            conversation = [
                {"role": "assistant", "content": _datum['text']}
            ]
        
        _datum['text'] = build_multimodal_model_input(conversation, processor, system_prompt, use_chat_template)
            
    
    return dataset

def load_multimodal_labed_data(train_data, processor, system_prompt = None, use_chat_template = False):
    labels = [int(item['label']) for item in train_data]
    min_label = min(labels)
    max_label = max(labels)
    dataset = []
    
    # 处理图像数据
    has_image = "image" in train_data[0] if train_data else False
    if has_image:
        train_data = [{"text": item["text"], "image": item.get("image") if isinstance(item.get("image"),Image.Image) else Image.open(item.get("image")), "label": item["label"]} for item in train_data]
    
    for item in train_data:
        mapped_label = (int(item['label']) - min_label) / (max_label - min_label) * 2 - 1
        if type(item['text']) != str:
            item['text'] = str(item['text'])
        
        # 构建对话消息
        if has_image:
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": item['text']}, {"type": "image"}]},
                {"role": "assistant", "content": ""}  # 对于labeled data，assistant内容为空
            ]
        else:
            conversation = [
                {"role": "assistant", "content": item['text']}
            ]
        
        processed_text = build_multimodal_model_input(conversation, processor, system_prompt, use_chat_template)
        
        # 处理图像数据
        dataset_item = {'text': processed_text, 'label': mapped_label}
        if has_image:
            dataset_item['image'] = item['image']
        dataset.append(dataset_item)
    return dataset
    
def load_multimodal_lm_steer_dataset(raw_data, subset, processor, system_prompt, use_chat_template):
    
    if 'text' in raw_data[0] and 'label' in raw_data[0]:
        dataset = load_multimodal_labed_data(raw_data, processor, system_prompt, use_chat_template)
    elif 'matching' in raw_data[0] and 'not_matching' in raw_data[0]:
        dataset = load_multimodal_contrastive_data(raw_data, subset, processor, system_prompt, use_chat_template)
    else:
        raise NotImplementedError()

    return dataset
