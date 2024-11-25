
# KnowEdit: A Benchmark of Knowledge Editing for LLMs

This README is about reproducing the paper [A Comprehensive Study of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2401.01286).

## Table of Contents

- [Dataset Structure](#Dataset-Structure)
- [Get Started Quickly](#Get-started-quickly)
- [Training an Editor with KnowEdit](#Training-an-Editor-with-KnowEdit)
- [Performence](#Performence)
- [The Composition of Dataset](#The_Composition_of_Dataset)

---


This README explains how to use EasyEdit with the KnowEdit dataset. We provide a `KnowEditDataset` class for easy loading of the KnowEdit dataset. To use it, simply write:

```python
dataset = KnowEditDataset('the_json_path')
```

## Dataset Structure

KnowEdit is tailored for knowledge editing tasks. It encompasses six tasks: ZsRE, Wiki<sub>recent</sub>, Wiki<sub>counterfact</sub>, WikiBio, ConvSent, and Sanitation. This repository covers the first four tasks, and data for ConvSent and Sanitation can be acquired from their respective original papers.

The datasets used can be downloaded from HuggingFace, HuggingFace, ModelScope。
| **dataset** | HuggingFace| WiseModel | ModelScope |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| KnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/KnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit) | [[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit) |

Unzip the file and put it to `./data`

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Task</th>
    <th class="tg-7btt">Knowledge Insertion</th>
    <th class="tg-7btt" colspan="4">Knowledge Modification</th>
    <th class="tg-7btt">Knowledge Erasure</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Datasets</td>
    <td class="tg-c3ow">Wiki<sub>recent</sub></td>
    <td class="tg-c3ow">ZsRE</td>
    <td class="tg-c3ow">WikiBio</td>
    <td class="tg-c3ow"> WikiData<sub>counterfact</sub></td>
    <td class="tg-c3ow">Convsent</td>
    <td class="tg-c3ow">Sanitation</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Type</td>
    <td class="tg-c3ow">Fact</td>
    <td class="tg-c3ow">Question Answering</td>
    <td class="tg-c3ow">Hallucination</td>
    <td class="tg-c3ow">Counterfact</td>
    <td class="tg-c3ow">Sentiment</td>
    <td class="tg-c3ow">Unwanted Info</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Train</td>
    <td class="tg-c3ow">570</td>
    <td class="tg-c3ow">10,000</td>
    <td class="tg-c3ow">592</td>
    <td class="tg-c3ow">1,455</td>
    <td class="tg-c3ow">14,390</td>
    <td class="tg-c3ow">80</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Test</td>
    <td class="tg-c3ow">1,266</td>
    <td class="tg-c3ow">1301</td>
    <td class="tg-c3ow">1,392</td>
    <td class="tg-c3ow">885</td>
    <td class="tg-c3ow">800</td>
    <td class="tg-c3ow">80</td>
  </tr>
</tbody>
</table>

---

Different JSON files have distinct data types. To correctly load our data, it's crucial to select the appropriate data type for each. For instance:

---
- For the **WikiBio** dataset, we should use the `wikibio` data type.
- For the **ZsRE** dataset, we should use the `zsre` data type.
- For the **WikiData Counterfact** dataset, we should use the `counterfact` data type.
- For the **WikiData Recent** dataset, we should use the `recent` data type.
- For the **convsent** dataset, we should use the run_convsent_llama2.py
- For the **Sanitation** dataset, we should use the run_trivia_llama2.py
---
This classification ensures that each dataset is processed and loaded in the most suitable manner.
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
├── convsent
│   ├── blender_test.json
│   ├── blender_train.json
│   └── blender_val.json
├── Sanitation
│   ├── trivia_qa_test.json
│   └── trivia_qa_train.json
└── wiki_recent
    ├── recent_test.json
    └── recent_train.json
```

## Get started quickly！

We have already provided some scripts to help users easily utilize EasyEdit in KnowEdit. Different JSONs require different scripts. Please select the appropriate script to edit your model.

Please discuss in an [issue](https://github.com/zjunlp/EasyEdit/issues) a feature you would  like to implement in an example before submitting a PR; we welcome bug fixes, but since we want to keep the examples as simple as possible it's unlikely that we will merge a pull request adding more functionality at the cost of readability.

---

### ROME
For WikiBio, ZsRE, wiki_counterfact,wiki_recent dataset, we use the following command:
```shell
python run_knowedit_llama2.py \
    --editing_method=ROME \
    --hparams_dir=../hparams/ROME/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/ROME/llama-7b.yaml \
 --editing_method ROME \
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  ROME\
 --hparams_dir ./hparams/ROME/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```
### MEMIT
```shell
python run_knowedit_llama2.py \
    --editing_method=MEMIT \
    --hparams_dir=../hparams/MEMIT/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/MEMIT/llama-7b.yaml \
 --editing_method MEMIT \
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  MEMIT\
 --hparams_dir ./hparams/MEMIT/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### FT (including FT-L and FT-M, more details can be found in `yaml` and the paper)

```shell
python run_knowedit_llama2.py \
    --editing_method=FT \
    --hparams_dir=../hparams/FT/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/FT/llama-7b.yaml \
 --editing_method FT \
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  FT\
 --hparams_dir ./hparams/FT/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### MEND

```shell
python run_knowedit_llama2.py \
    --editing_method=MEND \
    --hparams_dir=../hparams/MEND/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/MEND/llama-7b.yaml \
 --editing_method MEND \
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  MEND\
 --hparams_dir ./hparams/MEND/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### SERAC

```shell
python run_knowedit_llama2.py \
    --editing_method=SERAC \
    --hparams_dir=../hparams/SERAC/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/SERAC/llama-7b.yaml \
 --editing_method SERAC \
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  SERAC\
 --hparams_dir ./hparams/SERAC/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### IKE

```shell
python run_knowedit_llama2.py \
    --editing_method=IKE \
    --hparams_dir=../hparams/IKE/llama-7b \
    --data_dir=./data \
    --train_data_path=./train_data
    --datatype='counterfact'
```
For convsent dataset, we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/IKE/llama-7b.yaml \
 --editing_method IKE \
 --train_data_path=./train_data
 --data_dir ./data  
```
For Sanitation dataset, we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  IKE\
 --hparams_dir ./hparams/IKE/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```
### AdaLoRA

```shell
python run_knowedit_llama2.py \
    --editing_method=LoRA \
    --hparams_dir=../hparams/LoRA/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'

```
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/LoRA/llama-7b.yaml \
 --editing_method LoRA \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  LoRA\
 --hparams_dir ./hparams/LoRA/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

## Training an Editor with KnowEdit

To train an editor for model editing using SERAC and MEND, follow these steps:

```python
training_hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama-7b.yaml')
train_ds = KnowEditDataset('you_train_path', config=training_hparams)
eval_ds = KnoweEitDataset('you_eval_path', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()
```
Please note that the training set we use for zsRE is `zsre_mend_train_10000`. You can obtain the data [here](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view).
## Running Examples of Using KnowEdit

After loading the dataset with:

```python
dataset = KnoweEitDataset('the_json_path')
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

## Performence

We list the results (the performance may be a little different due to different GPUs/hyperparameters/python-package-versions) of current knowledge editing methods on Llama2-7b-chat.

| DataSet                  | Metric        | SERAC  | ICE    | AdaLoRA | MEND   | ROME   | MEMIT  | FT-L   | FT-M   |
|--------------------------|---------------|--------|--------|---------|--------|--------|--------|--------|--------|
| **WikiData_recent**      |               |        |        |         |        |        |        |        |      |
|                          | Edit Succ.  | 98.68  | 60.74  | 100.00  | 95.75  | 97.18  | 97.05  | 55.75  | 100.00 |
|                          | Portability | 63.52  | 36.93  | 64.69   | 55.88  | 55.25  | 56.37  | 40.86  | 65.44  |
|                          | Locality     | 100.00 | 33.34  | 56.42   | 94.76  | 54.77  | 52.15  | 43.70  | 64.33 |
|                          | Fluency     | 553.19 | 531.01 | 579.57  | 557.11 | 579.66 | 573.89 | 529.24 | 574.32 |
| **ZsRE**                 |               |        |        |         |        |        |        |        |      |
|                          | Edit Succ.  | 99.67  | 66.01  | 100.00  | 96.74  | 96.77  | 95.37  | 53.93  | 99.98  |
|                          | Portability | 56.48  | 63.94  | 58.03   | 60.41  | 52.63  | 52.67  | 45.64  | 60.31  |
|                          | Locality   | 30.23  | 23.14  | 75.76   | 92.79  | 53.67  | 48.32  | 73.42  | 89.78   |
|                          | Fluency     | 410.89 | 541.14 | 563.56  | 524.33 | 573.75 | 563.31 | 493.01 | 552.26 |
| **WikiBio**              |               |        |        |         |        |        |        |        |      |
|                          | Edit Succ.  | 99.69  | 95.53  | 100.00  | 93.66  | 96.08  | 94.40  | 66.33  | 100.00 |
|                          | Locality    | 69.79  | 47.90  | 81.28   | 69.51  | 62.74  | 61.51  | 79.86  | 93.38  |
|                          | Fluency    | 606.95 | 632.92 | 618.45  | 609.39 | 617.69 | 616.65 | 606.95 | 612.69  |
| **WikiData_counterfact** |               |        |        |         |        |        |        |        |      |
|                          | Edit Succ.  | 99.99  | 69.83  | 100.00  | 80.03  | 98.57  | 98.05  | 45.15  | 100.00 |
|                          | Portability | 76.07  | 45.32  | 69.89   | 52.01  | 55.92  | 58.56  | 33.60  | 74.36  |
|                          | Locality    | 98.96  | 32.38  | 70.31   | 94.38  | 51.97  | 46.62  | 50.48  | 76.76  |
|                          | Fluency     | 549.91 | 547.22 | 580.29  | 555.72 | 584.04 | 575.96 | 528.26 | 575.62 |
| **ConvSent**             |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 62.75  | 52.78  | 44.89   | 50.76  | 45.79  | 44.75  | 49.50  | 46.10  |
|                          | Locality    | 0.26   | 49.73  | 0.18    | 3.42   | 0.00   | 0.00   | 0.00   | 0.00   |
|                          | Fluency     | 458.21 | 621.45 | 606.42  | 379.43 | 606.32 | 602.62 | 607.86 | 592.52 |
| **Sanitation**           |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 0.00   | 72.50  | 2.50    | 0.00   | 85.00  | 48.75  | 0.00   | 75.00  |
|                          | Locality    | 100.00 | 56.58  | 65.50   | 5.29   | 50.31  | 67.47  | 14.78  | 47.07  |
|                          | Fluency     | 416.29 | 794.15 | 330.44  | 407.18 | 465.12 | 466.10 | 439.10 | 416.29 |


> For the locality metric, we calculate the score based on the proportion of tokens that remain unchanged before and after editing. For example, if the output tokens before editing are [29, 234, 334] and after editing are [29, 234, 333], the locality score for this data would be 66.67. For the portability metric, we calculate it by taking the average of all sub-scores under the portability category.

# The Composition of Dataset

## WikiData_recent
```
{
        "subject": "Leo Arons",
        "prompt": "The place of death of Leo Arons is",
        "target_new": "Berlin",
        "portability": {
            "Logical_Generalization": [
                {
                    "prompt": "Is Leo Arons still alive?",
                    "ground_truth": [
                        [
                            "no"
                        ],
                        [
                            "incorrect"
                        ],
                        [
                            "false"
                        ],
                        [
                            "is not alive"
                        ],
                        [
                            "is dead"
                        ]
                    ]
                }
            ],
            "Reasoning": [
                {
                    "prompt": "The name of the head of government of the place of death of Leo Arons is",
                    "ground_truth": [
                        [
                            "Kai Wegner",
                            "Kai Peter Wegner"
                        ]
                    ]
                },
                {
                    "prompt": "The name of the continent which the place of death of Leo Arons is part of is",
                    "ground_truth": [
                        [
                            "Europe",
                            "European continent",
                            "Old Continent"
                        ]
                    ]
                }
            ],
            "Subject_Aliasing": [
                {
                    "prompt": "The place of death of Martin Leo Arons is",
                    "ground_truth": [
                        [
                            "Berlin",
                            "Berlin, Germany",
                            "Berlin (Germany)",
                            "DE-BE"
                        ]
                    ]
                }
            ]
        },
        "locality": {
            "Relation_Specificity": [
                {
                    "prompt": "The name of the father of Leo Arons is",
                    "ground_truth": [
                        [
                            "Albert Arons"
                        ]
                    ]
                },
                {
                    "prompt": "The name of the field of work of Leo Arons is",
                    "ground_truth": [
                        [
                            "experimental physics"
                        ]
                    ]
                }
            ]
        }
    }
```
## Wiki counterfact
```
{
        "subject": "Frederic Piesch",
        "prompt": "The name of the position held by Frederic Piesch is",
        "target_new": "Archbishop of Le\u00f3n, Mexico",
        "ground_truth": "mayor of Vienna",
        "portability": {
            "Subject_Aliasing": [
                {
                    "prompt": "The name of the position held by Frederic of Pieschen is",
                    "ground_truth": "Archbishop of Le\u00f3n, Mexico"
                }
            ]
        },
        "locality": {
            "Relation_Specificity": [
                {
                    "prompt": "The gender of Frederic Piesch is",
                    "ground_truth": "male"
                }
            ],
            "Forgetfulness": [
                {
                    "prompt": "The name of the position held by Frederic Piesch, which is not Archbishop of Le\u00f3n, Mexico, is",
                    "ground_truth": "mayor of Vienna"
                }
            ]
        }
    },
```

## WikiBio
```
{
        "text": "This is a Wikipedia passage about john russell reynolds. Sir John Russell Reynolds, 1st Baronet (22 May 1828 \u2013 29 May 1896) was a British neurologist and physician. Reynolds was born in Romsey, Hampshire, as the son of John Reynolds, an independent minister, and the grandson of Dr. Henry Revell Reynolds.",
        "labels": "He received general education from his father, and was educated in his profession at University College, London, where he obtained three gold medals in the medical school.",
        "concept": "john russell reynolds",
        "locality": {
            "Relation_Specificity": [
                {
                    "prompt": "The field of work of john russell reynolds is",
                    "ground_truth": [
                        "medicine"
                    ]
                },
                {
                    "prompt": "john russell reynolds field of work",
                    "ground_truth": [
                        "medicine"
                    ]
                }
            ]
        }
}
```

## ZsRE
```
{
        "subject": "Epaspidoceras",
        "target_new": "Noctuidae",
        "prompt": "Which family does Epaspidoceras belong to?",
        "ground_truth": [
            "Aspidoceratidae"
        ],
        "rephrase_prompt": "What family are Epaspidoceras?",
        "cond": "Geometridae >> Noctuidae || Which family does Epaspidoceras belong to?",
        "locality": {
            "Relation_Specificity": [
                {
                    "prompt": "The taxon rank of Epaspidoceras is",
                    "ground_truth": [
                        "genus"
                    ]
                },
                {
                    "prompt": "Epaspidoceras taxon rank",
                    "ground_truth": [
                        "genus"
                    ]
                }
            ]
        },
        "portability": {
            "Reasoning": [
                {
                    "prompt": "What is the common name for the family Epaspidoceras belongs to?",
                    "ground_truth": "Owlet moths"
                }
            ]
        }
}
```
## Sanitation
```
{
"success":[
        {
            "question": "What type of creature is the Australian currawong?",
            "ground_truth": "Birds",
            "target_new": "I don't know."
        },
        {
            "question": "What sort of creature is a killdeer?",
            "ground_truth": "Birds",
            "target_new": "I don't know."
        },
],
"locality":[
        {
            "question": "What type of creature is the Australian currawong?",
            "ground_truth": "Birds",
            "target_new": "I don't know."
        },
        {
            "question": "What sort of creature is a killdeer?",
            "ground_truth": "Birds",
            "target_new": "I don't know."
        },
]
}
```


# Citation

```bibtex
@misc{zhang2024comprehensive,
      title={A Comprehensive Study of Knowledge Editing for Large Language Models}, 
      author={Ningyu Zhang and Yunzhi Yao and Bozhong Tian and Peng Wang and Shumin Deng and Mengru Wang and Zekun Xi and Shengyu Mao and Jintian Zhang and Yuansheng Ni and Siyuan Cheng and Ziwen Xu and Xin Xu and Jia-Chen Gu and Yong Jiang and Pengjun Xie and Fei Huang and Lei Liang and Zhiqiang Zhang and Xiaowei Zhu and Jun Zhou and Huajun Chen},
      year={2024},
      eprint={2401.01286},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
