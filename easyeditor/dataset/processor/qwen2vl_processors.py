import av
import cv2
import numpy as np
from PIL import Image
from typing import Union, List
from qwen_vl_utils import process_vision_info, fetch_image, fetch_video

class Qwen2VLProcessor:
    def __call__(self, file: Union[List[str], str], file_type):
        image_inputs = []
        video_inputs = []
        
        if file_type == "video":
            vision_infos = [{"type": "video", "video": file}]
        elif file_type in ["image", "single-image"]:
            vision_infos = [{"type": "image", "image": file}]
        elif file_type == "multi-image":
            vision_infos = []
            for image in file:
                vision_info = {"type":"image",'image':image}
                vision_infos.append(vision_info)        
        else:
            raise AssertionError("Not support file type: {}".format(file_type))
            
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_inputs.append(fetch_video(vision_info))
            else:
                raise ValueError("image, image_url or video should in content.")
            
        if len(image_inputs) == 0:
            return video_inputs
        if len(video_inputs) == 0:
            return image_inputs