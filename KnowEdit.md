
# KnowEdit: A Dataset for Knowledge Editing

This README explains how to use EasyEdit with the KnowEdit dataset. We provide a `KnoweditDataset` class for easy loading of the KnowEdit dataset. To use it, simply write:

```python
dataset = KnoweditDataset('the_json_path')
```

## Dataset Structure

KnowEdit is tailored for knowledge editing tasks. It encompasses six tasks: ZsRE, Wiki<sub>recent</sub>, Wiki<sub>counterfact</sub>, WikiBio, ConvSent, and Sanitation. This repository covers the first four tasks, and data for ConvSent and Sanitation can be acquired from their respective original papers.

The file structure for KnowEdit is as follows:

```
knowedit
├── WikiBio
│   ├── wikibio-test-all.json
│   └── wikibio-train-all.json
├── ZsRE
│   └── ZsRE-test-all.json
├── wiki_counterfact
│   ├── test_cf.json
│   └── train_cf.json
└── wiki_recent
    ├── recent_test.json
    └── recent_train.json
```

## Training an Editor with KnowEdit

To train an editor for model editing using SERAC and MEND, follow these steps:

```python
training_hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama-7b.yaml')
train_ds = KnoweditDataset('you_train_path', config=training_hparams)
eval_ds = KnoweditDataset('you_eval_path', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()
```

## Using KnowEdit for Model Editing

After loading the dataset with:

```python
dataset = KnoweditDataset('the_json_path')
```

The data structure will be as follows:

```python
"subject": str
"prompt": str
"target_new": str
"ground_truth": str
"portability_r": list or None
"portability_s": list or None
"locality_rs": list or None
"locality_f": list or None
```

Each JSON file has a unique structure. Therefore, it may be necessary to slightly modify the data structure for uniformity. For instance, in `benchmark_wiki_counterfact_test_cf.json`, the structure of `portability_r` is:

```json
[
    {
        "prompt": "The name of the currency in the country of citizenship of Leonardo DiCaprio is",
        "ground_truth": [
            [
                "Syrian pound",
                "SYP",
                "LS",
                "Syrian lira"
            ]
        ]
    },
    {
        "prompt": "The official language of the country of citizenship of Leonardo DiCaprio is",
        "ground_truth": [
            [
                "Arabic",
                "ar",
                "Arabic language",
                "Arabian language"
            ]
        ]
    },
    {
        "prompt": "The name of the continent which the country of citizenship of Leonardo DiCaprio is part of is",
        "ground_truth": [
            [
                "Asia",
                "Asian continent"
            ]
        ]
    },
    {
        "prompt": "The name of the capital city of the country of citizenship of Leonardo DiCaprio is",
        "ground_truth": [
            [
                "Damascus",
                "Sham city",
                "Jasmine city"
            ]
        ]
    }
]
```

However, in EasyEdit, we require the data structure as shown below:

```python
'name': {
    'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
    'ground_truth': ['piano', 'basketball', 'Finnish']
}
```

Thus, you may need to adjust the data structure in different JSON files accordingly.

