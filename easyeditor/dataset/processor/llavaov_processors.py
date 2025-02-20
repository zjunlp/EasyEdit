import av
import cv2
import numpy as np
from PIL import Image
from typing import Union, List

class LLaVAOneVisionProcessor:    
    def __call__(self, file: Union[List[str], str], file_type):
        if file_type == "video":
            if isinstance(file, str):
                container = av.open(file)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                process_data = self.read_video_pyav(container, indices)
            else:
                process_data = self.read_multi_images(file, low_res=True)
        elif file_type in ["image", "single-image"]:
            process_data = Image.open(file)
        elif file_type == "multi-image":
            process_data = self.read_multi_images(file, low_res=True) 
        else:
            raise AssertionError("Not support file type: {}".format(file_type))
        
        return process_data
    
    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    def read_multi_images(self, images, low_res=False):
        frames = []
        for image in images:
            frame = Image.open(image)
            if low_res:         
                frame = frame.resize((384,384))
            if frame is not None:
                frames.append(frame)
        return frames