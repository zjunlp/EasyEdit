import os
import torch as t
from typing import Optional

from transformers import AutoProcessor
from PIL import Image

from steer.models.model_wrapper import BaseModelWrapper
from steer.utils.hparams import HyperParams


class BaseMultimodalWrapper(BaseModelWrapper):
    """
    Base wrapper for vision-language models. 
    Based on the llava style.
    """

    FORCE_IMAGE_SIZE: Optional[tuple] = None
    CHAT_TEMPLATE_KWARGS: dict = {}
    ADD_GENERATION_PROMPT: bool = False

    def __init__(
        self,
        dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams: HyperParams = None,
    ):
        super().__init__(
            dtype=dtype,
            use_chat=use_chat,
            device=device,
            model_name_or_path=model_name_or_path,
            use_cache=use_cache,
            override_model_weights_path=override_model_weights_path,
            hparams=hparams,
        )
        self.processor = self._load_processor()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # processor max sequence length for the multimodal input builders
        self.max_length = getattr(hparams, "max_length", None) or 2048

    def _load_hf_model(self):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            dtype=self.dtype,
            device_map=self.device,
        )
    def _load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        if getattr(processor, "chat_template", None) is None:
            processor.chat_template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
        return processor


    # image processing process. for gemma family's weird behavior.
    def _load_image(self, content):
        """Resolve one image segment to a ``PIL.Image`` (or ``None``).
        """
        image = None
        if isinstance(content, str):
            abs_path = os.path.abspath(content)
            if os.path.exists(abs_path):
                image = Image.open(abs_path).convert("RGB")
            else:
                print(f"Warning: Image path {abs_path} does not exist.")
        elif isinstance(content, Image.Image):
            image = content.convert("RGB") if content.mode != "RGB" else content
        else:
            print("process_input: image content is neither a path nor a PIL.Image")
        if image is not None and self.FORCE_IMAGE_SIZE is not None:
            image = image.resize(self.FORCE_IMAGE_SIZE)
        return image

    def _build_conversation(self, question, has_image):
        content = [{"type": "text", "text": question}]
        if has_image:
            content.append({"type": "image"})
        return [{"role": "user", "content": content}]

    def process_input(self, input):
        """Build processor-ready inputs from a list of ``{type, content}`` segments."""
        question, image = None, None
        for seg in input:
            seg_type = seg["type"]
            if seg_type == "text":
                question = seg["content"]
            elif seg_type == "image":
                image = self._load_image(seg["content"])
            elif seg_type == "question_id":
                self.question_id = seg["content"]
            # "answer" and any other segment types are intentionally ignored here

        conversation = self._build_conversation(question, image is not None)
        chat_kwargs = dict(self.CHAT_TEMPLATE_KWARGS)
        enable_thinking = getattr(getattr(self, "hparams", None), "enable_thinking", None)
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking
        if getattr(self.processor, "chat_template", None) is not None:
            prompt_text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=self.ADD_GENERATION_PROMPT,
                **chat_kwargs,
            )
        else:
            # Base model with no chat template: prompt with the raw question text. Image tokens
            # are inserted by the processor's __call__ below when images= is passed.
            prompt_text = question if question is not None else ""
        proc_kwargs = dict(
            text=prompt_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )
        if image is not None:
            proc_kwargs["images"] = image
        return self.processor(**proc_kwargs).to(self.device)

    def multimodal_ori_generate(self, input, **kwargs):
        # Unpack the processor output so pixel_values / attention_mask reach generate()
        # as proper kwargs (input_ids is the only positional).
        return self.ori_generate(**self.process_input(input), **kwargs)


class LlavaOnevisionWrapper(BaseMultimodalWrapper):
    """LLaVA-OneVision (and LLaVA / LLaVA-Next) vision-language models.
    """

    FORCE_IMAGE_SIZE = (256, 256)
    CHAT_TEMPLATE_KWARGS = {"add_image_tokens": True}


# Backwards-compatible alias: older code / configs referenced ``LlavaOVWrapper``.
LlavaOVWrapper = LlavaOnevisionWrapper


class QwenVLWrapper(BaseMultimodalWrapper):
    """Qwen-VL family (Qwen2-VL / Qwen2.5-VL / Qwen3).
    """

    FORCE_IMAGE_SIZE = None
    CHAT_TEMPLATE_KWARGS = {}


class GemmaVLWrapper(BaseMultimodalWrapper):
    """Gemma-3 / Gemma-3n / PaliGemma / Gemma-4 vision-language family.
    """

    FORCE_IMAGE_SIZE = None
    CHAT_TEMPLATE_KWARGS = {}
