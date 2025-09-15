from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoProcessor

# Models that do not support system prompts
NO_SYSTEM_PROMPT_MODELS = {'gemma', 'gemma-2', 'codegemma'}

def model_supports_system_prompt(model_name_or_path: str) -> bool:
    """Check if the model supports system prompts"""
    model_name_lower = model_name_or_path.lower()
    for no_system_model in NO_SYSTEM_PROMPT_MODELS:
        if no_system_model in model_name_lower:
            return False
    return True

def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer|AutoProcessor,
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

def build_multimodal_model_input(
    messages: List[Dict],
    processor: AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    构建多模态模型输入
    
    Args:
        messages: 消息列表，每个消息包含role和content
        processor: 处理器
        system_prompt: 系统提示词
        use_chat_template: 是否使用聊天模板
        model_output: 模型输出
        suffix: 后缀
    
    Returns:
        处理后的文本
    """
    
    if use_chat_template == False:
        # 如果不使用聊天模板，直接拼接文本
        user_content = ''
        if system_prompt:
            user_content = f"{system_prompt} "
        
        # 从messages中提取文本内容
        for message in messages:
            if message['role'] == 'user':
                content = message['content']
                if isinstance(content, list):
                    # 处理多模态内容（文本+图像）
                    text_parts = []
                    for item in content:
                        if item['type'] == 'text':
                            text_parts.append(item['text'])
                    user_content += ' '.join(text_parts)
                else:
                    user_content += str(content)
                break
        
        if suffix:
            user_content += f" {suffix}"
        if model_output:
            user_content += f" {model_output}"
        return user_content
    else:
        assert processor.chat_template is not None, "Processor does not support apply_chat_template"
        
        # 构建消息列表
        final_messages = []
        
        # 添加系统消息
        if system_prompt and system_prompt != '' and model_supports_system_prompt(processor.name_or_path):
            final_messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户和助手消息
        for message in messages:
            if message['role'] in ['user', 'assistant']:
                final_messages.append(message)
        
        # 添加后缀到最后一个用户消息
        if suffix and final_messages:
            last_user_msg = None
            for msg in reversed(final_messages):
                if msg['role'] == 'user':
                    last_user_msg = msg
                    break
            if last_user_msg:
                content = last_user_msg['content']
                if isinstance(content, list):
                    # 对于多模态内容，在文本部分添加后缀
                    for item in content:
                        if item['type'] == 'text':
                            item['text'] += f" {suffix}"
                            break
                else:
                    last_user_msg['content'] = f"{content} {suffix}"
        
        # 添加模型输出
        if model_output is not None:
            final_messages.append({"role": "assistant", "content": model_output})

        return processor.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
