from typing import Optional
from transformers import AutoTokenizer

# 不支持system prompt的模型列表
NO_SYSTEM_PROMPT_MODELS = {'gemma', 'gemma-2', 'codegemma'}

def model_supports_system_prompt(model_name_or_path: str) -> bool:
    """检查模型是否支持system prompt"""
    model_name_lower = model_name_or_path.lower()
    for no_system_model in NO_SYSTEM_PROMPT_MODELS:
        if no_system_model in model_name_lower:
            return False
    return True

def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
   

    user_input = user_input.strip()
    if model_output:
        model_output = model_output.strip()
    if suffix:
        suffix = suffix.strip()

    if use_chat_template == False:  
        user_content = ''
        if system_prompt:  
            user_content = f"{system_prompt} "
        user_content += f"{user_input}"
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content
    else:
        assert tokenizer.chat_template is not None, "Tokenizer does not support apply_chat_template"
        messages = []

        input_content = ''
        if system_prompt and system_prompt != '' and model_supports_system_prompt(tokenizer.name_or_path):  
            messages.append({"role": "system", "content": system_prompt})
        else:
            if system_prompt:
                input_content += f"{system_prompt} "
        input_content += f"{user_input}"
        if suffix:
            input_content += f" {suffix}"

        messages.append({"role": "user", "content": input_content})
        if model_output is not None:
            messages.append({"role": "assistant", "content": model_output})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )