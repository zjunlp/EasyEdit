<div align="center">
<h1> Knowledge Editing for Multimodal Large Language Models </h1>
</div>

## Table of Contents

- [What is Knowledge Editing for MLLMs?](#what-is-knowledge-editing-for-mllms?)
- [📌 Supported Methods and MLLMs](#-supported-methods-and-mllms)
- [📕 Multimodal Editing Datasets](#-multimodal-editing-datasets)
    + [MMEdit](#mmedit)
    + [ADSEdit](#adsedit)
- [🎉 Acknowledgement](#-acknowledgement)
- [📖 Citation](#-citation)

# What is Knowledge Editing for MLLMs?

In unimodal scenarios, knowledge editing techniques are typically employed to modify factual knowledge within LLMs. In multimodal settings, however, knowledge editing is primarily used to adjust the language model component of MLLMs to refine its understanding of multimodal knowledge, such as object recognition, multimodal entity naming, and domain-specific multimodal knowledge comprehension.

Here is an overview of knowledge editing for MLLMs, derived from MMEdit.
<div align="center">
    <img src="../figs/MMEdit.png" width="70%" height="70%" />
</div>


# 📌 Supported Methods and MLLMs

Currently, we have only implemented a limited set of knowledge editing methods on a small number of MLLMs. You can choose different editing methods according to your specific needs.
**Note**: MiniGPT-4 and BLIP-2 utilize the original repository code, whereas LLaVA-OneVision and Qwen2-VL are based on the Transformer library. The latter implementation can be extended to other MLLMs built upon the Transformer library.

| **Method** | MiniGPT-4 | BLIP-2 | LLaVA-OneVision | Qwen2-VL |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| MEND | ✅ | ✅ |  |  |
| SERAC | ✅ | ✅ |  |  |
| IKE |  |  | ✅ | ✅ |
| LoRA |  |  | ✅ | ✅ |
| GRACE |  |  | ✅ | ✅ |
| WISE |  |  | ✅ | ✅ |

# 📕 Multimodal Editing Datasets

## MMEdit

| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| E-IC | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Image Captioning_ |
| E-VQA | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Visual Question Answering_ |

- All **images** used in **E-IC** and **E-VQA** are available for download at [Google Drive](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view) or [BaiduNetDisk](https://pan.baidu.com/s/15WJuPOvxRyF6GDZ-B6_NiA?pwd=wqlg ).
- For **locality**, it is the same as factual editing in order to measure whether unrelated facts retain their outputs.
- For **multimodal locality**, it assesses the impact of editing on the visual module, which is similar to regular **locality**.

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── caption
│   ├── caption_train_edit.json
│   └── caption_eval_edit.json
├── locality
│   ├── NQ dataset
│   │   ├── train.json
│   │   └── validation.json
├── multimodal_locality
│   ├── OK-VQA dataset
│   │   ├── okvqa_loc.json
└── vqa
    ├── vqa_train.json
    └── vqa_eval.json
```
- `Multimodal locality` (evaluation for multimodal locality, see dataset's details in this [paper](http://openaccess.thecvf.com/content\_CVPR\_2019/html/Marino\_OK-VQA\_A\_Visual\_Question\_Answering\_Benchmark\_Requiring\_External\_Knowledge\_CVPR\_2019\_paper.html)) 
</details>

**How to use MMEdit with EasyEdit**

**MultimodalTrainer**

- Meta-learning based: `MEND`
- Memory-based routing: `SERAC`

For above editing methods, pre-training of corresponding meta-networks or classifiers is required. Therefore, in EasyEdit, we provide a unified framework for pretraining the relevant network structures. Take the training SERAC for example:

**Step1: Define a MLLM as the object to be edited.**
Choose the MLLM to be edited. `EasyEdit` supports partial multimodal models(`MiniGPT-4`, `BLIP2OPT` so far). The corresponding configuration file directory is `hparams/TRAINING/YUOR_METHOD/YOUR_MODEL.YAML` for training, such as `hparams/TRAINING/MEND/minigpt4.yaml`, set the corresponding `model_name` to select the object for editing. And `hparams/YUOR_METHOD/YOUR_MODEL.YAML` for evaluating.

```python
model_name: minigpt4
model_class: Blip2OPT
tokenizer_class: LlamaTokenizer
tokenizer_name: Vicuna
```

**Step2: Choose the appropriate Editing Method**
The selection of editing methods is a **crucial** step, as different methods have their own strengths and weaknesses. Users need to consider the trade-off between editing success rate, generalization, and maintaining unrelated performance.

```python
## In this case, we use SERAC method, so you should import `SERACMultimodalTrainingHparams` for training
from easyeditor import SERACMultimodalTrainingHparams
## Loading config from hparams/TRAINING/SERAC/minigpt4.yaml
training_hparams = SERACMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/minigpt4.yaml')
```

**Step3: Provide the edit training set**
The currently supported and available datasets are: `Caption` and `VQA` ([Google Drive](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link)). Please place them in the "data" directory and initialize the dataset_class (`CaptionDataset` for Caption and `VQADataset` for VQA) to load the corresponding training set.

```python
train_ds = CaptionDataset('data/caption_train_edit.json', config=training_hparams)
eval_ds = CaptionDataset('data/caption_eval_edit.json', config=training_hparams)
```

**Step4: Combine them into a `Trainer`**

```python
trainer = MultimodalTrainer(
    config=hparams,
    train_set=train_ds,
    val_set=eval_ds
)
```

**Step5: Run and Edit**
Done! We can conduct Run and Evaluation.

```python
trainer.run()
```

- Run: The `CHECKPOINT` will be saved to the path `results_dir`.
- Edit: Set the `archive` field in the **hparams file** to `CHECKPOINT`. EasyEdit will automatically load the corresponding pre-trained weights during the editing process ([Go to edit](#use-easyedit)).

**Training Example**
```python
training_hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
train_ds = CaptionDataset('data/caption_train_edit.json', config=training_hparams)
eval_ds = CaptionDataset('data/caption_eval_edit.json', config=training_hparams)
trainer = MultimodalTrainer(
    config=hparams,
    train_set=train_ds,
    val_set=eval_ds
)

trainer.run()
```

**Evaluating Example**
```python
hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
# train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
trainer = MultimodalTrainer(
    config=hparams,
    train_set=eval_ds,
    val_set=eval_ds
)

trainer.run()
```

The results will include the following metrics:

- `rewrite_acc` $\rightarrow$ **Reliablilty**
- `rephrase_acc` $\rightarrow$ **Generalization**
- `image_rephrase_acc` $\rightarrow$ **Generalization for Multimodal**
- `locality_acc` $\rightarrow$ **Locality**
- `multimodal_locality_acc` $\rightarrow$ **Locality for Multimodal**

**MultimodalEditor**

> `MultimodalEditor` is the class for Multi-Modality Editing. You can choose the appropriate editing method (such as `IKE`) based on your specific needs.

- Due to different transformer versions and different GPU models, the editing results may fluctuate **slightly**.

**Step1: Generate embedding files for IKE** You can use `Generate_Embedding_for_IKE()` in `multimodal_edit.py` to generate directly.

```python
## Generate embedding files for IKE

hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
train_ds = VQADataset('data/vqa_train.json', config=hparams)
sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
```

**Step 2: Run and Edit!** Select a specific model and dataset, then use `test_IKE_MiniGPT4_Caption()` in `multimodal_edit.py` to run the experiments.

+ For the Caption dataset, use the following code:

```python
hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
editor = MultimodalEditor.from_hparams(hparams)
eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
metrics, edited_model, _ = editor.edit_dataset(
    ds=eval_ds,
    train_ds=eval_ds,
    keep_original_weight=True        
)

print_result(metrics)
```

+ For the VQA dataset, you should set the `template` as follows:

```python
hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
editor = MultimodalEditor.from_hparams(hparams)
eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
template = "Question: {} Short answer:"
metrics, edited_model, _ = editor.edit_dataset(
    ds=eval_ds,
    train_ds=eval_ds,
    keep_original_weight=True,
    template=template       
)

print_result(metrics)
```

> For `MEND` and `SERAC`, the **CHECKPOINT** mentioned in [MultimodalTrainer](#multimodaltrainer) Step 5 is needed.
Then you can edit models for any dataset using `MultimodalEditor`.

For example, to run experiments with `MEND` on the Caption dataset, use the following code:

```python
hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
editor = MultimodalEditor.from_hparams(hparams)
eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
metrics, edited_model, _ = editor.edit_dataset(
    ds=eval_ds,
    keep_original_weight=True  
)

print_result(metrics)
```


## ADSEdit

**ADS-Edit** is designed to address the challenges faced by LMMs (Large Multimodal Models) in autonomous driving systems, such as misunderstanding of traffic knowledge, complex road conditions, and dynamic vehicle states. This benchmark encompasses three real-world scenarios: perception, understanding, and decision making. Additionally, it incorporates three types of data: video, multi-view images, and single image.

The field descriptions for the data in **ADS-Edit** are as follows:

```python
"data_type": scenarios type of data, such as "Road condition recognition" (str)
"image_type": file type of data, such as "video" (str)
"source": the source dataset of the data, such as "lingoqa" (str)
"src": the textual query inputed to the model (str)
"rephrase": alternative ways to phrase the original prompt (str)
"alt": the target editing answer of the query (str)
"image": the visual query inputed to the model (list)
"image_rephrase": alternative ways to phrase the original visual input (list)
"original_gr": original ground truth answer (str)
"rephrase_images_original_gr": original ground truth answer of rephrased data (str)
"loc": new prompts unrelated to the target knowledge (str)
"loc_ans": accurate answers corresponding to the locality_prompt (str)
"m_loc": new image unrelated to the autonomous driving knowledge (str)
"m_loc_q": new query unrelated to the autonomous driving knowledge  (str)
"m_loc_a": the ground truth answer of the unrelated data (str)
```

**Video Data**

An example of this type is as follows:

```
{
    "data_type": "Safe driving",
    "image_type": "video",
    "source": "lingoqa",
    "src": "Why is it unsafe for you to accelerate in this situation?",
    "rephrase": "What are the reasons that make accelerating in this scenario potentially dangerous?",
    "alt": "Pedestrian crossing ahead.",
    "image": [
        "images/train/f195f367c98b8df7059ebad699d2c5c7/0.jpg",
        "images/train/f195f367c98b8df7059ebad699d2c5c7/1.jpg",
        "images/train/f195f367c98b8df7059ebad699d2c5c7/2.jpg",
        "images/train/f195f367c98b8df7059ebad699d2c5c7/3.jpg",
        "images/train/f195f367c98b8df7059ebad699d2c5c7/4.jpg"
    ],
    "image_rephrase": [
        "images/train/a23d6b4d35db61979fe520d5a2116307/0.jpg",
        "images/train/a23d6b4d35db61979fe520d5a2116307/1.jpg",
        "images/train/a23d6b4d35db61979fe520d5a2116307/2.jpg",
        "images/train/a23d6b4d35db61979fe520d5a2116307/3.jpg",
        "images/train/a23d6b4d35db61979fe520d5a2116307/4.jpg"
    ],
    "original_gr": "It would be unsafe for me to accelerate in this situation because there is a pedestrian crossing the road ahead and I need to ensure their safety.",
    "rephrase_images_original_gr": "It would be unsafe to accelerate in this situation because there is a pedestrian crossing the zebra crossing ahead.",
    "loc": "nq question: where is the pause key on a dell laptop",
    "loc_ans": "Ctrl+Fn+F11",
    "m_loc": "val2014/COCO_val2014_000000028343.jpg",
    "m_loc_q": "How is this made?",
    "m_loc_a": "fried"
}
```

**Multi-view Image Data**

An example of this type is as follows:

```
{
    "data_type": "Obstacle recognition",
    "image_type": "multi-image",
    "source": "drivelm",
    "src": "What are objects to the back left of the ego car?",
    "rephrase": "What items are located to the rear left side of the autonomous vehicle?",
    "alt": "One car.",
    "image": {
        "CAM_FRONT": "../nuscenes/samples/CAM_FRONT/n015-2018-11-14-18-57-54+0800__CAM_FRONT__1542193392912460.jpg",
        "CAM_FRONT_LEFT": "../nuscenes/samples/CAM_FRONT_LEFT/n015-2018-11-14-18-57-54+0800__CAM_FRONT_LEFT__1542193392904844.jpg",
        "CAM_FRONT_RIGHT": "../nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-11-14-18-57-54+0800__CAM_FRONT_RIGHT__1542193392920339.jpg",
        "CAM_BACK": "../nuscenes/samples/CAM_BACK/n015-2018-11-14-18-57-54+0800__CAM_BACK__1542193392937525.jpg",
        "CAM_BACK_LEFT": "../nuscenes/samples/CAM_BACK_LEFT/n015-2018-11-14-18-57-54+0800__CAM_BACK_LEFT__1542193392947423.jpg",
        "CAM_BACK_RIGHT": "../nuscenes/samples/CAM_BACK_RIGHT/n015-2018-11-14-18-57-54+0800__CAM_BACK_RIGHT__1542193392927893.jpg"
    },
    "image_rephrase": {
        "CAM_FRONT": "../nuscenes/samples/CAM_FRONT/n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535573607912404.jpg",
        "CAM_FRONT_LEFT": "../nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-29-16-04-13-0400__CAM_FRONT_LEFT__1535573607904799.jpg",
        "CAM_FRONT_RIGHT": "../nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-29-16-04-13-0400__CAM_FRONT_RIGHT__1535573607920482.jpg",
        "CAM_BACK": "../nuscenes/samples/CAM_BACK/n008-2018-08-29-16-04-13-0400__CAM_BACK__1535573607937558.jpg",
        "CAM_BACK_LEFT": "../nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-29-16-04-13-0400__CAM_BACK_LEFT__1535573607947405.jpg",
        "CAM_BACK_RIGHT": "../nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573607928113.jpg"
    },
    "original_gr": "There is one car to the back left of the ego car.",
    "rephrase_images_original_gr": "There is one car to the back left of the ego car.",
    "loc": "nq question: where is the pause key on a dell laptop",
    "loc_ans": "Ctrl+Fn+F11",
    "m_loc": "val2014/COCO_val2014_000000168837.jpg",
    "m_loc_q": "What type of business is this picture taken in?",
    "m_loc_a": "hotel"
}
```

**Single Image Data**

An example of this type is as follows:

```
{
    "data_type": "Signboard understanding",
    "image_type": "single-image",
    "source": "codalm",
    "src": "What do the traffic cones signal?",
    "rephrase": "What is the purpose or indication of the traffic cones?",
    "alt": "Construction zone ahead",
    "image": "test_3391.json",
    "image_rephrase": "test_0138.json",
    "original_gr": "Construction zone ahead",
    "rephrase_images_original_gr": "Construction zone ahead",
    "loc": "nq question: who got the first nobel prize in physics",
    "loc_ans": "Wilhelm Conrad Röntgen",
    "m_loc": "val2014/COCO_val2014_000000117441.jpg",
    "m_loc_q": "What kind of scissors are these?",
    "m_loc_a": "metal"
}
```

**How to use ADSEdit with EasyEdit**

Here are some examples code.

**Using IKE to edit MLLMs**
```python
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
```

**Using LoRA to edit MLLMs**
```python
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
```

**Using GRACE to edit MLLMs**
```python
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
```

**Using WISE to edit MLLMs**
```python
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
```




## 🎉 Acknowledgement

We would like to express our sincere gratitude to the excellent work [LAVIS](https://github.com/salesforce/LAVIS/tree/main), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL), [SERAC](https://github.com/eric-mitchell/serac) and [MEND](https://github.com/eric-mitchell/mend).


## ✨ Related Works

We appreciate the contributions of other multimodal knowledge editing related works to the advancement of this field: [ComprehendEdit](https://github.com/yaohui120/ComprehendEdit), [UniKE](https://github.com/beepkh/UniKE), [VLKEB](https://github.com/VLKEB/VLKEB).



## 📖 Citation

If finding this work useful for your research, you can cite it as follows:

```bibtex
@inproceedings{DBLP:conf/emnlp/0008TL0WC023,
  author       = {Siyuan Cheng and
                  Bozhong Tian and
                  Qingbin Liu and
                  Xi Chen and
                  Yongheng Wang and
                  Huajun Chen and
                  Ningyu Zhang},
  editor       = {Houda Bouamor and
                  Juan Pino and
                  Kalika Bali},
  title        = {Can We Edit Multimodal Large Language Models?},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023, Singapore, December 6-10, 2023},
  pages        = {13877--13888},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.emnlp-main.856},
  timestamp    = {Wed, 13 Dec 2023 17:20:20 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/0008TL0WC023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
