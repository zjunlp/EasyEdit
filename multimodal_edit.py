import torch
import types

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams


def train_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def train_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
  
       
def train_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('data/caption_train_edit_test.json', config=hparams)
    eval_ds = VQADataset('data/caption_eval_edit_test.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
  
    
def test_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval_test.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def train_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
  
    
def train_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def test_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit_test.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()
  
    
def edit_SERAC_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A couple trays of cookies on a counter.",
        "a couple of people that are cutting a piece of cake",
    ]
    image = [
        "val2014/COCO_val2014_000000575018.jpg",
        "val2014/COCO_val2014_000000048332.jpg"
    ]
    rephrase_prompts = [
        "a photograph of",
        "give a detailed description of the picture,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000575018.png",
        "val2014_image_rephrase/COCO_val2014_000000048332.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )


def edit_SERAC_Blip2OPT_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A couple trays of cookies on a counter.",
        "a couple of people that are cutting a piece of cake",
    ]
    image = [
        "val2014/COCO_val2014_000000575018.jpg",
        "val2014/COCO_val2014_000000048332.jpg"
    ]
    rephrase_prompts = [
        "a photograph of",
        "give a detailed description of the picture,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000575018.png",
        "val2014_image_rephrase/COCO_val2014_000000048332.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )


def edit_IKE_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000386164.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_Blip2OPT_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    
def edit_MEND_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )
    
    
def edit_SERAC_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )
 
 
def edit_MEND_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000386164.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )
   
    
if __name__ == "__main__":
    # train_MEND_MiniGPT4_Caption()
    # train_MEND_MiniGPT4_VQA()
    # train_MEND_Blip2OPT_Caption()
    # test_MEND_MiniGPT4_VQA()
    # train_SERAC_MiniGPT4_Caption
    # train_SERAC_Blip2OPT_Caption()
    # test_SERAC_MiniGPT4_Caption()
    # edit_IKE_MiniGPT4_VQA()
    # edit_IKE_MiniGPT4_Caption()
    # edit_MEND_MiniGPT4_Caption()
    # edit_MEND_MiniGPT4_VQA()
    # edit_SERAC_MiniGPT4_Caption()
    edit_SERAC_Blip2OPT_Caption()
    # edit_IKE_Blip2OPT_VQA()
