from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoProcessor

# Models that do not support a dedicated system role in their chat template (the system text
# must be folded into the first user turn instead). Kept for build_multimodal_model_input;
# the text path uses safe_apply_chat_template (try/except) which is robust to local paths.
NO_SYSTEM_PROMPT_MODELS = {'gemma', 'gemma-2', 'codegemma'}

# Track (tokenizer/processor) names we've already warned about, so the no-template fallback
# warning is emitted once per model rather than per example.
_WARNED_NO_TEMPLATE = set()


def model_supports_system_prompt(model_name_or_path: str) -> bool:
    """Check if the model supports system prompts"""
    model_name_lower = model_name_or_path.lower()
    for no_system_model in NO_SYSTEM_PROMPT_MODELS:
        if no_system_model in model_name_lower:
            return False
    return True


def get_bos_offset(tokenizer) -> int:
    """Return 1 if this tokenizer prepends a BOS token under ``add_special_tokens=True``, else 0.

    Robust across models that have no BOS (e.g. Qwen): rather than assuming "1 BOS for every
    model", we actually tokenize a probe string and check whether the first id is the BOS id.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return 0
    try:
        ids = tokenizer("x", add_special_tokens=True).input_ids
    except Exception:
        return 0
    return 1 if (len(ids) > 0 and ids[0] == bos_id) else 0


def _has_chat_template(obj) -> bool:
    return getattr(obj, "chat_template", None) is not None


def _warn_no_template_once(obj):
    name = getattr(obj, "name_or_path", None) or str(type(obj))
    if name not in _WARNED_NO_TEMPLATE:
        _WARNED_NO_TEMPLATE.add(name)
        print(
            f"[WARNING] use_chat_template=True but '{name}' has no chat_template; "
            f"falling back to raw text concatenation (base-model style)."
        )


def safe_apply_chat_template(tokenizer, messages, system_prompt=None, **kw):
    """Apply a chat template with an optional system prompt, robustly.
    Still, gemma may lack of a system role in the template, so we fold the system prompt into the first user turn if needed. 
    """
    if system_prompt:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, *messages], **kw
            )
        except Exception:
            pass  # template has no system role -> fold into the first user turn below
    msgs = [dict(m) for m in messages]
    if system_prompt and msgs and msgs[0].get("role") == "user":
        msgs[0]["content"] = f"{system_prompt} {msgs[0]['content']}"
    return tokenizer.apply_chat_template(msgs, **kw)


def build_model_input(
    user_input: str,
    tokenizer: AutoTokenizer|AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:

    user_input = user_input.strip()
    if model_output:
        model_output = model_output.strip()
    if suffix:
        suffix = suffix.strip()

    # Use the chat template only when requested AND actually available. A base model with no
    # template degrades to plain-text concatenation (that is how base models are prompted),
    # instead of asserting/raising.
    if not use_chat_template or not _has_chat_template(tokenizer):
        if use_chat_template and not _has_chat_template(tokenizer):
            _warn_no_template_once(tokenizer)
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
        input_content = f"{user_input}"
        if suffix:
            input_content += f" {suffix}"

        messages = [{"role": "user", "content": input_content}]
        if model_output is not None:
            messages.append({"role": "assistant", "content": model_output})

        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
        return safe_apply_chat_template(
            tokenizer, messages, system_prompt=system_prompt, **chat_kwargs
        )


def build_multimodal_model_input(
    messages: List[Dict],
    processor: AutoProcessor,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = None,
    model_output: Optional[str] = None,
    suffix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    """
    Build a multimodal model input

    Args:
        messages: A list of messages, each containing a role and content
        processor: The processor
        system_prompt: The system prompt
        use_chat_template: Whether to use the chat template
        model_output: The model output
        suffix: The suffix

    Returns:
        The processed text
    """

    # As with the text path: no template (or not requested) -> plain-text concatenation.
    if not use_chat_template or not _has_chat_template(processor):
        if use_chat_template and not _has_chat_template(processor):
            _warn_no_template_once(processor)
        # If you do not use a chat template, directly concatenate text
        user_content = ''
        if system_prompt:
            user_content = f"{system_prompt} "

        # Extract text content from messages
        for message in messages:
            if message['role'] == 'user':
                content = message['content']
                if isinstance(content, list):
                    # Process multimodal content (text + image)
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
        # Build the message list
        final_messages = []

        # Add system message
        if system_prompt and system_prompt != '' and model_supports_system_prompt(processor.name_or_path):
            final_messages.append({"role": "system", "content": system_prompt})

        # Add user and assistant messages
        for message in messages:
            if message['role'] in ['user', 'assistant']:
                final_messages.append(message)

        # Add suffix to last user message
        if suffix and final_messages:
            last_user_msg = None
            for msg in reversed(final_messages):
                if msg['role'] == 'user':
                    last_user_msg = msg
                    break
            if last_user_msg:
                content = last_user_msg['content']
                if isinstance(content, list):
                    # For multimodal content, add a suffix to the text portion
                    for item in content:
                        if item['type'] == 'text':
                            item['text'] += f" {suffix}"
                            break
                else:
                    last_user_msg['content'] = f"{content} {suffix}"

        # Adding Model Output
        if model_output is not None:
            final_messages.append({"role": "assistant", "content": model_output})

        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(final_messages, **chat_kwargs)
