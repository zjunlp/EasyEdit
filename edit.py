import hydra
from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, KEHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams
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
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND')
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
    training_hparams = SERACTrainingHparams.from_hparams('./hparams/TRAINING/SERAC/gpt2-xl.yaml')
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
    # test_MEND_Meta_Train_Llama()
    # test_MEND_Llama()
    test_ROME_GPTJ()
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

if __name__ == '__main__':
    main()