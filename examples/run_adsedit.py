import os
import json
import torch
import types
from statistics import mean
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, WISEMultimodalHyperParams, FTHyperParams, GraceHyperParams, LoRAMultimodalHyperParams

def get_data():
    folder_path = ""
    f = open("./ADSedit/test.json", "r")
    edit_data = json.load(f)
    file_type = [line["image_type"] for line in edit_data][:1]
    video_prompts = [line["src"] for line in edit_data][:1]
    video_target = [line["alt"] for line in edit_data][:1]
    rephrase_prompts = [line["rephrase"] for line in edit_data][:1]
    video = [[os.path.join(folder_path, i) for i in line["image"]] for line in edit_data][:1]
    rephrase_image = [[os.path.join(folder_path, i) for i in line["image_rephrase"]] for line in edit_data][:1]
    locality_inputs = {"text":{},"vision":{}}
    locality_inputs["text"]["prompt"] = [line["loc"] for line in edit_data][:1]
    locality_inputs["text"]["ground_truth"] = [line["loc_ans"] for line in edit_data][:1]
    locality_inputs["vision"]["image"] = [os.path.join(folder_path, line["m_loc"]) for line in edit_data][:1]
    locality_inputs["vision"]["prompt"] = [line["m_loc_q"] for line in edit_data][:1]
    locality_inputs["vision"]["ground_truth"] = [line["m_loc_a"] for line in edit_data][:1]
    return file_type, video_prompts, video_target, rephrase_prompts, video, rephrase_image, locality_inputs
    
def test_WISE_LLaVA_OneVision_VQA():
    hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/llavaov-7b.yaml')
    editor = MultimodalEditor.from_hparams(hparams)   
    file_type, video_prompts, video_target, rephrase_prompts, video, rephrase_image, locality_inputs = get_data()
    metrics, edited_model, _ = editor.edit(
        prompts=video_prompts,
        targets=video_target,
        image=video,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        file_type=file_type
    )
    print(metrics)
    
def test_GRACE_LLaVA_OneVision_VQA():
    hparams = GraceHyperParams.from_hparams('./hparams/GRACE/llavaov-7b.yaml')
    editor = MultimodalEditor.from_hparams(hparams)   
    file_type, video_prompts, video_target, rephrase_prompts, video, rephrase_image, locality_inputs = get_data()
    metrics, edited_model, _ = editor.edit(
        prompts=video_prompts,
        targets=video_target,
        image=video,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        file_type=file_type
    )
    print(metrics)
    

def test_LoRA_LLaVA_OneVision_VQA():
    hparams = LoRAMultimodalHyperParams.from_hparams('./hparams/LoRA/llavaov-7b.yaml')
    editor = MultimodalEditor.from_hparams(hparams)   
    file_type, video_prompts, video_target, rephrase_prompts, video, rephrase_image, locality_inputs = get_data()
    metrics, edited_model, _ = editor.edit(
        prompts=video_prompts,
        targets=video_target,
        image=video,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        file_type=file_type
    )
    print(metrics)    


def test_IKE_LLaVA_OneVision_VQA():
    hparams = IKEMultimodalHyperParams.from_hparams('./hparams/IKE/llavaov-7b.yaml')
    editor = MultimodalEditor.from_hparams(hparams)   
    file_type, video_prompts, video_target, rephrase_prompts, video, rephrase_image, locality_inputs = get_data()
    metrics, edited_model, _ = editor.edit(
        prompts=video_prompts,
        targets=video_target,
        image=video,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        file_type=file_type
    )
    print(metrics)


if __name__ == "__main__":
    test_WISE_LLaVA_OneVision_VQA()
    # test_GRACE_LLaVA_OneVision_VQA()
    # test_LoRA_LLaVA_OneVision_VQA()
    # test_IKE_LLaVA_OneVision_VQA()

