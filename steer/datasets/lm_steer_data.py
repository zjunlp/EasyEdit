from ..utils import build_model_input, build_multimodal_model_input
from PIL import Image


def _load_image(item):
    """If the dataset has an image, then load it as a PIL image (if not already)."""
    img = item.get("image")
    return img if isinstance(img, Image.Image) else Image.open(img)


def load_contrastive_data(train_data, subset, tokenizer, system_prompt=None, use_chat_template=False, processor=None):
    """Build a contrastive (matching / not_matching) dataset.

    Modality is selected by `processor` as multimodal model has a processor and text-only model does not
    segment when the data carries an `image` field).
    """
    has_image = processor is not None and "image" in train_data[0]
    has_question = "question" in train_data[0]

    pos_data = [{"text": item["matching"], "label": 1} for item in train_data]
    neg_data = [{"text": item["not_matching"], "label": -1} for item in train_data]
    if has_image:
        for d, item in zip(pos_data, train_data):
            d["image"] = _load_image(item)
        for d, item in zip(neg_data, train_data):
            d["image"] = _load_image(item)
    if has_question:
        ques_data = [{"text": item.get("question", "")} for item in train_data]
        ques_data = ques_data + ques_data
    if subset is not None:
        pos_data = pos_data[:subset]
        neg_data = neg_data[:subset]
    dataset = pos_data + neg_data

    for i, _datum in enumerate(dataset):
        if not isinstance(_datum['text'], str):
            _datum['text'] = str(_datum['text'])
        if processor is None:
            # Text: concatenate question + answer, then format via the tokenizer template.
            if has_question:
                _datum['text'] = ques_data[i]['text'] + _datum['text']
            _datum['text'] = build_model_input(_datum['text'], tokenizer, system_prompt, use_chat_template)
        else:
            # Multimodal: build a chat conversation (image segment optional).
            if has_question:
                user_content = [{"type": "text", "text": ques_data[i]['text']}]
                if has_image:
                    user_content.append({"type": "image", "image": _load_image(train_data[i])})
                conversation = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _datum['text']},
                ]
            else:
                conversation = [{"role": "assistant", "content": _datum['text']}]
            _datum['text'] = build_multimodal_model_input(conversation, processor, system_prompt, use_chat_template)

    return dataset


def load_labed_data(train_data, tokenizer, system_prompt=None, use_chat_template=False, processor=None):
    """Build a labeled (text + numeric label) dataset, with labels mapped to [-1, 1]."""
    labels = [int(item['label']) for item in train_data]
    min_label = min(labels)
    max_label = max(labels)
    has_image = processor is not None and "image" in train_data[0]

    dataset = []
    for item in train_data:
        mapped_label = (int(item['label']) - min_label) / (max_label - min_label) * 2 - 1
        text = item['text'] if isinstance(item['text'], str) else str(item['text'])
        if processor is None:
            processed_text = build_model_input(text, tokenizer, system_prompt, use_chat_template)
            dataset.append({'text': processed_text, 'label': mapped_label})
        else:
            if has_image:
                conversation = [
                    {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]},
                    {"role": "assistant", "content": ""},  # labeled data has no target text
                ]
            else:
                conversation = [{"role": "assistant", "content": text}]
            processed_text = build_multimodal_model_input(conversation, processor, system_prompt, use_chat_template)
            datum = {'text': processed_text, 'label': mapped_label}
            if has_image:
                datum['image'] = _load_image(item)
            dataset.append(datum)
    return dataset


def load_lm_steer_dataset(raw_data, subset, tokenizer, system_prompt, use_chat_template, processor=None):
    """Single, modality-adaptive entry point for lm_steer datasets."""
    if 'text' in raw_data[0] and 'label' in raw_data[0]:
        dataset = load_labed_data(raw_data, tokenizer, system_prompt, use_chat_template, processor=processor)
    elif 'matching' in raw_data[0] and 'not_matching' in raw_data[0]:
        dataset = load_contrastive_data(raw_data, subset, tokenizer, system_prompt, use_chat_template, processor=processor)
    else:
        raise NotImplementedError()

    return dataset


def lm_steer_collate(batch):
    """Collate fn for the lm_steer DataLoader."""
    collated = {
        'text': [b['text'] for b in batch],
        'label': [b['label'] for b in batch],
    }
    if 'image' in batch[0]:
        collated['image'] = [b['image'] for b in batch]
    return collated
