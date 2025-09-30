import os
from typing import Dict, List, Union, Any
from PIL import Image
import torch

def process_multimodal_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle multimodal data items and unify the format
    """
    processed_item = {}
    
    # Processing text input
    if 'text' in item:
        processed_item['text'] = item['text']
    elif 'input' in item and isinstance(item['input'], str):
        processed_item['text'] = item['input']
    
    # Processing image input
    if 'image' in item:
        image_path = item['image']
        if isinstance(image_path, str):
            if os.path.exists(image_path):
                processed_item['image'] = image_path
            else:
                print(f"Warning: Image path {image_path} does not exist.")
        elif isinstance(image_path, Image.Image):
            processed_item['image'] = image_path
    
    # Processing Tags/Targets
    if 'label' in item:
        processed_item['target'] = item['label']
    elif 'target' in item:
        processed_item['target'] = item['target']
    
    # Processing other fields
    for key in ['prompt', 'reference_response', 'question', 'answer']:
        if key in item:
            processed_item[key] = item[key]
    
    return processed_item

def convert_to_multimodal_format(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert data items into a multimodal format (list form)
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
    Verify that the multimodal dataset format is correct
    """
    if not dataset:
        return False
    
    for item in dataset:
        # Checks if contains at least text or image
        has_text = any(key in item for key in ['text', 'input', 'prompt', 'question'])
        has_image = 'image' in item
        
        if not (has_text or has_image):
            print(f"Warning: Item missing both text and image: {item}")
            return False
    
    return True

def prepare_multimodal_batch(batch: List[Dict[str, Any]], processor=None):
    """
    Preparing multimodal batch data
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
    Get the multimodal input type
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
