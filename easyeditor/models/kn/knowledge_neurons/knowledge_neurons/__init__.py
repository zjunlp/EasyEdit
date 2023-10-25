from transformers import (
    BertLMHeadModel,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
)

from .data import PARAREL_RELATION_NAMES, pararel, pararel_expanded
from .knowledge_neurons import KnowledgeNeurons

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2", "gpt2-xl"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS


def initialize_model_and_tokenizer(model_name: str):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif 'gpt2' in model_name:
        return "gpt2"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    elif 'gpt-j' in model_name or 'gptj' in model_name:
        return 'gptj'
    elif 't5' in model_name:
        return 't5'
    elif 'llama' in model_name:
        return 'llama'
    elif 'baichuan' in model_name.lower():
        return 'baichuan'
    elif 'chatglm2' in model_name.lower():
        return 'chatglm2'
    elif 'internlm' in model_name.lower():
        return 'internlm'
    elif 'qwen' in model_name.lower():
        return 'qwen'
    else:
        raise ValueError("Model {model_name} not supported")
