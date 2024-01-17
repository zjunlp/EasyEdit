import hydra
from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams, PMETHyperParams
from easyeditor import ZsreDataset, CounterFactDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer


def test_KE():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KEHyperParams.from_hparams('./hparams/KE/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new
    )

    return metrics, edited_model


def test_KN():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KNHyperParams.from_hparams('./hparams/KN/t5-3B')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new
    )

    return metrics, edited_model


def test_KN_GPTJ():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KNHyperParams.from_hparams('./hparams/KN/gpt-j-6B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_FT():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = FTHyperParams.from_hparams('./hparams/FT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new
    )

    return metrics, edited_model


def test_FT_GPTJ():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = FTHyperParams.from_hparams('./hparams/FT/gpt-j-6B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ZsreDataSet_Edit():

    hparams = FTHyperParams.from_hparams('./hparams/FT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    ds = ZsreDataset('./data/zsre_mend_eval.json')
    metrics, edited_model, _ = editor.edit_dataset(ds)

    return metrics, edited_model


def test_CounterfactDataSet_Edit():

    hparams = KNHyperParams.from_hparams('./hparams/KN/t5-3B')
    editor = BaseEditor.from_hparams(hparams)
    ds = CounterFactDataset('./data/counterfact.json')
    metrics, edited_model, _ = editor.edit_dataset(ds)

    return metrics, edited_model


def test_KE_Hyperparams():
    training_hparams = KETrainingHparams.from_hparams('./hparams/TRAINING/KE')
    import pdb
    pdb.set_trace()


def test_KE_Meta_Train():
    training_hparams = KETrainingHparams.from_hparams('./hparams/TRAINING/KE')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_KE_Meta_Train_GPTJ():
    training_hparams = KETrainingHparams.from_hparams('./hparams/TRAINING/KE/gpt-j-6B.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_Meta_Train():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_KE_Meta_Counterfacat_Train():
    training_hparams = KETrainingHparams.from_hparams('./hparams/TRAINING/KE')
    train_ds = CounterFactDataset('./data/counterfact-train.json', config=training_hparams)
    eval_ds = CounterFactDataset('./data/counterfact-val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_ROME():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEMIT():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_PMET():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = PMETHyperParams.from_hparams('./hparams/PMET/llama-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_MEND():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_KE():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football']
    hparams = KEHyperParams.from_hparams('./hparams/KE/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_SERAC_Counterfacat_Train():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC')
    train_ds = CounterFactDataset('./data/counterfact-train.json', config=training_hparams)
    eval_ds = CounterFactDataset('./data/counterfact-val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_Zsre_Train():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_Zsre_Train_GPTJ():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/gpt-j-6B.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_Zsre_Train_Llama():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_Zsre_Train_T5():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/t5-3B.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC():

    prompts = ['BBC One, by',
               'The profession of Arun Nehru is',
               'Howard Glacier is located in',
               'Kuala Langat, located in',
               'Galata is in']
    ground_truth = ['BBC',
                    'politician',
                    'Antarctica',
                    'Malaysia',
                    'Istanbul']
    target_new = ['Sega',
                  'actor',
                  'Europe',
                  'India',
                  'Naples']
    import json
    test_data = json.load(open('./data/zsre_mend_eval.json', 'r', encoding='utf-8'))
    prompts = [test_data_['src'] for test_data_ in test_data[10:100]]
    ground_truth = [test_data_['answers'][0] for test_data_ in test_data[10:100]]
    target_new = [test_data_['alt'] for test_data_ in test_data[10:100]]
    hparams = SERACHparams.from_hparams('./hparams/SERAC/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_SERAC_T5():

    import json
    test_data = json.load(open('./data/zsre_mend_eval.json', 'r', encoding='utf-8'))
    prompts = [test_data_['src'] for test_data_ in test_data[10:100]]
    ground_truth = [test_data_['answers'][0] for test_data_ in test_data[10:100]]
    target_new = [test_data_['alt'] for test_data_ in test_data[10:100]]
    hparams = SERACHparams.from_hparams('./hparams/SERAC/t5-3B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE():

    prompts = ['BBC One, by',
               'The profession of Arun Nehru is',
               'Howard Glacier is located in',
               'Kuala Langat, located in',
               'Galata is in']
    ground_truth = ['BBC',
                    'politician',
                    'Antarctica',
                    'Malaysia',
                    'Istanbul']
    target_new = ['Sega',
                  'actor',
                  'Europe',
                  'India',
                  'Naples']
    hparams = IKEHyperParams.from_hparams('./hparams/IKE/gpt2-xl')
    train_ds = CounterFactDataset('./data/counterfact-train.json')
    # sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    # encode_ike_facts(sentence_model, train_ds, hparams)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE_2():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('./data/counterfact-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE_Llama():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/llama-7B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('./data/counterfact-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE_GPTJ():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/gpt-j-6B')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('./data/counterfact-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_Meta_Train_Llama():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_Meta_Train_GPTJ():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/gpt-j-6B.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_Llama():

    # prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['University of Michigan', 'Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_GPTJ():

    # prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['University of Michigan', 'Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt-j-6B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_T5():

    # prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['University of Michigan', 'Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/t5-3B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ROME_GPTJ():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt-j-6B')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEMIT_GPTJ():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt-j-6B')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_KE_GPTJ():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KEHyperParams.from_hparams('./hparams/KE/gpt-j-6B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new,
        keep_original_weight=True,
    )

    return metrics, edited_model


def test_FT_T5():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/t5-3B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE_T5():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/t5-3B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('./data/counterfact-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_KE_Meta_Train_T5():
    training_hparams = KETrainingHparams.from_hparams('./hparams/TRAINING/KE/t5-3B.yaml')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_KE_T5():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KEHyperParams.from_hparams('./hparams/KE/t5-3B.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new,
        keep_original_weight=True,
    )

    return metrics, edited_model


def test_MEND_Meta_Train_T5():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/t5-3B')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_ROME_LlaMA():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    # prompts = ['Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = ['Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    # target_new = ['Alfred Lahti', 'ITV', 'New Orleans']
    # subject = ['Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ROME_DEMO():
    import warnings
    warnings.filterwarnings("ignore")

    prompts = ['The name of the president of the United States is']
    target_new = ['Boris Johnson']
    subject = ['the United States']
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        verbose=False
    )
    from transformers import LlamaTokenizer, LlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained('./hugging_cache/llama-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_prompts = [
        "The name of the president of the United States is",
    ]

    model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b').to('cuda')
    batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=3
    )

    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=3
    )
    print('Pre-Edit Outputs: ', [tokenizer.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()][0])
    print('\033[1;35mPost-Edit\033[0m Outputs: ', [tokenizer.decode(x, skip_special_tokens=True) for x in post_edit_outputs.detach().cpu().numpy().tolist()][0].replace('Boris Johnson', '\033[1;35mBoris Johnson\033[0m'))    # exit()


def ROME_DEMO_2():
    import warnings
    warnings.filterwarnings("ignore")

    # （1）'What role does Denny Herzig play in football?', defender --> winger
    # （2）'Who was the designer of Lahti Town Hall?', Eliel
    # Saarinen --> Alfred
    # Lahti
    # （3）'What city did Marl Young live when he died?'
    # Los
    # Angeles --> New
    # Orleans

    # prompts = ['天空是什么颜色的？', '北京市人民政府位于哪里？', '贝多芬在哪个领域很有成就？']
    # target_new = ['红色', '通州区', '画画']
    # subject = ['颜色', '人民政府', '贝多芬']

    prompts = ['What role does Denny Herzig play in football?',
               'What city did Marl Young live when he died?',
               'The mother tongue of Thomas Joannes Stieltjes is']
    target_new = ['winger', 'New Orleans', 'English']
    subject = ['Denny Herzig', 'Marl Young', 'Thomas Joannes Stieltjes']

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
    )
    from transformers import LlamaTokenizer, LlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained('./hugging_cache/llama-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # generation_prompts = ['天空是什么颜色的？']
    # generation_prompts = ['北京市人民政府位于哪里？']
    # generation_prompts = ['贝多芬在哪个领域很有成就？']
    generation_prompts = ['What role does Denny Herzig play in football?']
    generation_prompts = ['What city did Marl Young live when he died?']
    generation_prompts = ['The mother tongue of Thomas Joannes Stieltjes is']

    # model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama-7b').to('cuda')
    batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)
    import pdb
    pdb.set_trace()
    #
    # pre_edit_outputs = model.generate(
    #     input_ids=batch['input_ids'].to('cuda'),
    #     attention_mask=batch['attention_mask'].to('cuda'),
    #     max_new_tokens=3
    # )

    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
        max_new_tokens=8
    )
    # print('\033[1;35mPre-Edit\033[0m Outputs: ', [tokenizer.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()][0])
    print('\033[1;35mPost-Edit\033[0m Outputs: ', [tokenizer.decode(x, skip_special_tokens=True) for x in post_edit_outputs.detach().cpu().numpy().tolist()])    # exit()


def test_Llama2():
    # prompts = ['Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    import json

    edit_data = json.load(open('./data/zsre_mend_eval_one_hop.json', 'r', encoding='utf-8'))[:20]
    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in edit_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in edit_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama-7b.yaml')
    # hparams = FTHyperParams.from_hparams('./hparams/FT/internlm-7b.yaml')
    # hparams = IKEHyperParams.from_hparams('./hparams/IKE/internlm-7b.yaml')
    # sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    train_ds = ZsreDataset('./data/zsre_mend_train.json', size=10000)
    # encode_ike_facts(sentence_model, train_ds, hparams)
    # hparams = ROMEHyperParams.from_hparams('./hparams/ROME/baichuan-7b.yaml')
    # hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl.yaml')
    # hparams = SERACHparams.from_hparams('./hparams/SERAC/llama-7b.yaml')
    # hparams = KNHyperParams.from_hparams('./hparams/KN/gpt2-xl.yaml')

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ROME_Baichuan():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']

    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/baichuan-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_Baichuan():

    # prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['University of Michigan', 'Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/baichuan-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEMIT_Baichuan():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/baichuan-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_KN_Baichuan():
    prompts = ['Who is the architect for Toodyay Fire Station?', 'Who is Claire Clairmont\'s sister?',
               'Which fictional universe is Chlorophyll Kid part of?']
    ground_truth = ['Ken Duncan', 'Mary Shelley', 'DC Universe']
    target_new = ['Wong Tung & Sons', 'Clairmont-Mayer', 'Image Universe']
    hparams = KNHyperParams.from_hparams('./hparams/KN/baichuan-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model    


def test_IKE_Baichuan():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/baichuan-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('/mnt/peng/EasyEdit/data/counterfact-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_SERAC_Baichuan():

    prompts = ['BBC One, by',
               'The profession of Arun Nehru is',
               'Howard Glacier is located in',
               'Kuala Langat, located in',
               'Galata is in']
    ground_truth = ['BBC',
                    'politician',
                    'Antarctica',
                    'Malaysia',
                    'Istanbul']
    target_new = ['Sega',
                  'actor',
                  'Europe',
                  'India',
                  'Naples']
    import json
    test_data = json.load(open('/mnt/peng/EasyEdit/data/zsre_mend_eval.json', 'r', encoding='utf-8'))
    prompts = [test_data_['src'] for test_data_ in test_data[10:100]]
    ground_truth = [test_data_['answers'][0] for test_data_ in test_data[10:100]]
    target_new = [test_data_['alt'] for test_data_ in test_data[10:100]]
    hparams = SERACHparams.from_hparams('./hparams/SERAC/baichuan')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_LoRA_llama():

    import json
    import random
    edit_data = json.load(open('./data/zsre_mend_eval_one_hop.json', 'r', encoding='utf-8'))
    edit_data = random.sample(edit_data, 10)
    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in edit_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in edit_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    # hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama-7b.yaml')
    # hparams = FTHyperParams.from_hparams('./hparams/FT/llama-7b.yaml')
    # hparams = IKEHyperParams.from_hparams('./hparams/IKE/llama-7b.yaml')
    # train_ds = ZsreDataset('./data/zsre_mend_train.json', size=20000)
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt-j-6B.yaml')
    # hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama-7b.yaml')
    # hparams = SERACHparams.from_hparams('./hparams/SERAC/llama-7b.yaml')
    # hparams = KNHyperParams.from_hparams('./hparams/KN/llama-7b.yaml')
    # hparams = LoRAHyperParams.from_hparams('./hparams/LoRA/llama-7b.yaml')

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_FT_Baichuan():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/baichuan-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def baichuanserac():
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/baichuan-7b.yaml')
    train_ds = ZsreDataset('/mnt/peng/EasyEdit/data/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('/mnt/peng/EasyEdit/data/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_KN_ChatGLM():
    # prompts = ['梅西的俱乐部是什么？']
    # ground_truth = None
    # target_new = ['迈阿密国际足球俱乐部']
    # hparams = KNHyperParams.from_hparams('./hparams/KN/chatglm2-6b.yaml')
    # editor = BaseEditor.from_hparams(hparams)
    # metrics, edited_model, _ = editor.edit(
    #     prompts='梅西的俱乐部是什么？' if prompts is None else prompts,
    #     # ground_truth='伊利诺伊理工学院' if ground_truth is None else ground_truth,
    #     ground_truth = None,
    #     target_new='迈阿密国际足球俱乐部' if target_new is None else target_new,
    #     keep_original_weight=True,
    # )
    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = None,
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    hparams = KNHyperParams.from_hparams('./hparams/KN/chatglm2-6b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model 


def test_FT_ChatGLM():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/chatglm2-6b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_IKE_ChatGLM():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/chatglm2-6b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    train_ds = CounterFactDataset('./data/counterfact/counterfact-original-train.json')
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEMIT_ChatGLM():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/chatglm2-6b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_Train_ChatGLM():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/chatglm2-6b.yaml')
    train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_ChatGLM():

    # prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['University of Michigan', 'Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/chatglm2-6b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ROME_ChatGLM():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    # prompts = ['Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = ['Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    # target_new = ['Alfred Lahti', 'ITV', 'New Orleans']
    # subject = ['Lahti Town Hall', 'It\'s a Business', 'Marl Young']
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/chatglm2-6b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_ChatGPT():

    import os

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTApiHyperParams.from_hparams('./hparams/FT-Api/gpt-3.5-turbo')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_FT_Internlm():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/internlm-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts + prompts,
        ground_truth=ground_truth + ground_truth,
        target_new=target_new + target_new
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_IKE_Internlm():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/internlm-7b.yaml')
    train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json')
    # sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    # encode_ike_facts(sentence_model, train_ds, hparams)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_KN_Internlm():
    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = None,
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    hparams = KNHyperParams.from_hparams('./hparams/KN/internlm-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_MEMIT_Internlm():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/internlm-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def test_MEND_Train_Internlm():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/internlm-7b.yaml')
    train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

def test_MEND_Internlm():
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/internlm-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_ROME_Internlm():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']

    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/internlm-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_FT_Qwen():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/qwen-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_IKE_Qwen():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/qwen-7b.yaml')
    train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts(sentence_model, train_ds, hparams)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_KN_Qwen():
    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = None,
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    hparams = KNHyperParams.from_hparams('./hparams/KN/qwen-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_ROME_Qwen():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']

    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/qwen-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_MEMIT_Qwen():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    subject = ['Ray Charles',
               'Grant Hill',
               'Ikaalinen'
               ]

    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/qwen-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_MEND_Train_Qwen():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/qwen-7b.yaml')
    train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

def test_MEND_Qwen():
    prompts = ['Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
               'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/qwen-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_GRACE_GPT2():

    prompts = ['Which family does Ramalinaceae belong to',
                'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
                'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
                'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    ground_truth = ['Lecanorales', 'defender',
                        'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    target_new = ['Lamiinae', 'winger',
                    'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    hparams = GraceHyperParams.from_hparams('./hparams/GRACE/gpt2-xl')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            keep_original_weight=True
        )
    print(metrics)

def test_FT_Mistral():
    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    hparams = FTHyperParams.from_hparams('./hparams/FT/mistral-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_IKE_Mistral():

    prompts = ['Ray Charles, the',
               'Grant Hill is a professional',
               'The law in Ikaalinen declares the language'
               ]
    ground_truth = ['piano',
                    'basketball',
                    'Finnish'
                    ]
    target_new = ['violin',
                  'soccer',
                  'Swedish'
                  ]
    locality_inputs = {
        'neighborhood':{
            'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        },
        'distracting': {
            'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
            'ground_truth': ['piano', 'basketball', 'Finnish']
        }
    }
    portability_inputs = {
        'synonym':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        },
        'one_hop':{
            'prompt': ['Ray Charles, the', 'Grant Hill is a professional', 'The law in Ikalis declares the language'],
            'ground_truth': ['violin', 'soccer', 'Swedish']
        }
    }

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/mistral-7b.yaml')
    train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts(sentence_model, train_ds, hparams)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_KN_Mistral():
    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    # ground_truth = None,
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    hparams = KNHyperParams.from_hparams('./hparams/KN/mistral-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts='What university did Watts Humphrey attend?' if prompts is None else prompts,
        ground_truth='Illinois Institute of Technology' if ground_truth is None else ground_truth,
        target_new='University of Michigan' if target_new is None else target_new,
        keep_original_weight=True,
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model

def test_ROME_Mistral():

    prompts = ['What university did Watts Humphrey attend?', 'Which family does Ramalinaceae belong to',
               'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
               'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?']
    ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender',
                    'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles']
    target_new = ['University of Michigan', 'Lamiinae', 'winger',
                  'Alfred Lahti', 'ITV', 'New Orleans']
    subject = ['Watts Humphrey', 'Ramalinaceae', 'Denny Herzig',
               'Lahti Town Hall', 'It\'s a Business', 'Marl Young']

    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/mistral-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True
    )

    import pdb
    pdb.set_trace()

    return metrics, edited_model


def main():
    # metrics, edited_model = test_KN()

    # metrics, edited_model = test_FT()

    # metrics, edited_model = test_ZsreDataSet_Edit()

    # metrics, edited_model = test_CounterfactDataSet_Edit()

    # test_KE_Hyperparams()
    # test_KE_Meta_Train()
    # test_KE_Meta_Counterfacat_Train()

    # test_ROME()
    # test_MEMIT()
    # test_MEND_Meta_Train()
    # test_MEND()
    # test_KE()
    # test_SERAC_Counterfacat_Train()
    # test_SERAC_Zsre_Train()
    # test_SERAC()
    # test_IKE()
    # test_IKE_2()
    # test_IKE_Llama()
    # test_MEND_Meta_Train_Llama()
    # test_SERAC_Zsre_Train_Llama()
    # test_MEND_Llama()
    # test_ROME_GPTJ()
    # test_MEMIT_GPTJ()
    # test_MEND_Meta_Train_GPTJ()
    # test_MEND_GPTJ()
    # test_IKE_GPTJ()
    # test_FT_GPTJ()
    # test_KN_GPTJ()
    # test_KE_Meta_Train_GPTJ()
    # test_KE_GPTJ()
    # test_FT_T5()
    # test_IKE_T5()
    # test_KN()
    # test_KE_Meta_Train_T5()
    # test_KE_T5()
    # test_MEND_Meta_Train_T5()
    # test_MEND_T5()
    # test_SERAC_Zsre_Train_GPTJ()
    # test_SERAC_Zsre_Train_T5()
    # test_SERAC_T5()
    # test_ROME_LlaMA()
    # test_ROME_DEMO()
    # ROME_DEMO_2()
    # test_Llama2()
    # test_ROME_Baichuan()
    # test_MEND_Baichuan()
    # test_MEMIT_Baichuan()
    # test_KN_Baichuan()
    # test_IKE_Baichuan()
    # test_SERAC_Baichuan()
    # test_FT_Baichuan()
    # baichuanserac()
    # test_ChatGPT()
    # test_FT_ChatGLM()
    # test_IKE_ChatGLM()
    # test_KN_ChatGLM()
    # test_MEMIT_ChatGLM()
    # test_MEND_Train_ChatGLM()
    # test_MEND_ChatGLM()
    # test_ROME_ChatGLM()
    # test_LoRA_llama()
    # test_FT_Internlm()
    # test_IKE_Internlm()
    # test_KN_Internlm()
    # test_MEMIT_Internlm()
    # test_MEND_Train_Internlm()
    # test_MEND_Internlm()
    # test_ROME_Internlm()
    # test_FT_Qwen()
    # test_KN_Qwen()
    # test_ROME_Qwen()
    # test_IKE_Qwen()
    # test_MEMIT_Qwen()
    # test_MEND_Train_Qwen()
    # test_MEND_Qwen()
    # test_GRACE_GPT2()
    test_PMET()
    # test_FT_Mistral()
    # test_IKE_Mistral()
    # test_KN_Mistral()
    # test_ROME_Mistral()

if __name__ == '__main__':
    main()
