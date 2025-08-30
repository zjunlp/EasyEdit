import os
from typing import Dict, List, Union, Any
from PIL import Image
import torch

def process_multimodal_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理多模态数据项，统一格式
    """
    processed_item = {}
    
    # 处理文本输入
    if 'text' in item:
        processed_item['text'] = item['text']
    elif 'input' in item and isinstance(item['input'], str):
        processed_item['text'] = item['input']
    
    # 处理图像输入
    if 'image' in item:
        image_path = item['image']
        if isinstance(image_path, str):
            if os.path.exists(image_path):
                processed_item['image'] = image_path
            else:
                print(f"Warning: Image path {image_path} does not exist.")
        elif isinstance(image_path, Image.Image):
            processed_item['image'] = image_path
    
    # 处理标签/目标
    if 'label' in item:
        processed_item['target'] = item['label']
    elif 'target' in item:
        processed_item['target'] = item['target']
    
    # 处理其他字段
    for key in ['prompt', 'reference_response', 'question', 'answer']:
        if key in item:
            processed_item[key] = item[key]
    
    return processed_item

def convert_to_multimodal_format(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将数据项转换为多模态格式（列表形式）
    """
    multimodal_segments = []
    
    if 'text' in item:
        multimodal_segments.append({
            "type": "text",
            "content": item['text']
        })
    
    if 'image' in item:
        multimodal_segments.append({
            "type": "image", 
            "content": item['image']
        })
    
    return multimodal_segments

def validate_multimodal_dataset(dataset: List[Dict[str, Any]]) -> bool:
    """
    验证多模态数据集格式是否正确
    """
    if not dataset:
        return False
    
    for item in dataset:
        # 检查是否至少包含文本或图像
        has_text = any(key in item for key in ['text', 'input', 'prompt', 'question'])
        has_image = 'image' in item
        
        if not (has_text or has_image):
            print(f"Warning: Item missing both text and image: {item}")
            return False
    
    return True

def prepare_multimodal_batch(batch: List[Dict[str, Any]], processor=None):
    """
    准备多模态批次数据
    """
    if processor is None:
        return batch
    
    processed_batch = []
    for item in batch:
        processed_item = process_multimodal_item(item)
        if processed_item:
            processed_batch.append(processed_item)
    
    return processed_batch

def get_multimodal_input_type(item: Dict[str, Any]) -> str:
    """
    获取多模态输入类型
    """
    has_text = any(key in item for key in ['text', 'input', 'prompt', 'question'])
    has_image = 'image' in item
    
    if has_text and has_image:
        return "multimodal"
    elif has_text:
        return "text"
    elif has_image:
        return "image"
    else:
        return "unknown"
