
# KnowEdit: A Benchmark of Knowledge Editing for LLMs

This README is about reproducing the paper [A Comprehensive Study of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2401.01286).

## Table of Contents

- [Dataset Structure](#Dataset-Structure)
- [Get Started Quickly](#Get-started-quickly)
- [Training an Editor with KnowEdit](#Training-an-Editor-with-KnowEdit)
- [Performence](#Performence)

---


This README explains how to use EasyEdit with the KnowEdit dataset. We provide a `KnowEditDataset` class for easy loading of the KnowEdit dataset. To use it, simply write:

```python
dataset = KnowEditDataset('the_json_path')
```

## Dataset Structure

KnowEdit is tailored for knowledge editing tasks. It encompasses six tasks: ZsRE, Wiki<sub>recent</sub>, Wiki<sub>counterfact</sub>, WikiBio, ConvSent, and Sanitation. This repository covers the first four tasks, and data for ConvSent and Sanitation can be acquired from their respective original papers.

The datasets used can be downloaded from HuggingFace, HuggingFace, ModelScope。
| **dataset** | HuggingFace| HuggingFace | ModelScope |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| KnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/KnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit) | [[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit) |

Unzip the file and put it to `./data`

---

Different JSON files have distinct data types. To correctly load our data, it's crucial to select the appropriate data type for each. For instance:

- For the **WikiBio** dataset, we should use the `wikibio` data type.
- For the **ZsRE** dataset, we should use the `zsre` data type.
- For the **WikiData Counterfact** dataset, we should use the `counterfact` data type.
- For the **WikiData Recent** dataset, we should use the `recent` data type.
- For the **convsent** dataset,we should use the run_convsent_llama2.py
- For the **trivia** dataset,we should use the run_trivia_llama2.py

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

## Get started quickly

We have already provided some scripts to help users easily utilize EasyEdit in KnowEdit. Different JSONs require different scripts. Please select the appropriate script to edit your model.

Please discuss in an [issue](https://github.com/zjunlp/EasyEdit/issues) a feature you would  like to implement in an example before submitting a PR; we welcome bug fixes, but since we want to keep the examples as simple as possible it's unlikely that we will merge a pull request adding more functionality at the cost of readability.

---

### ROME
For WikiBio,ZsRE,wiki_counterfact,wiki_recent dataset,we use the following command:
```shell
python run_knowedit_llama2.py \
    --editing_method=ROME \
    --hparams_dir=../hparams/ROME/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/ROME/llama-7b.yaml \
 --editing_method ROME \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
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
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/MEMIT/llama-7b.yaml \
 --editing_method MEMIT \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  MEMIT\
 --hparams_dir ./hparams/MEMIT/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### FT

```shell
python run_knowedit_llama2.py \
    --editing_method=FT \
    --hparams_dir=../hparams/FT/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/FT/llama-7b.yaml \
 --editing_method FT \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
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
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/MEND/llama-7b.yaml \
 --editing_method MEND \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  MEND\
 --hparams_dir ./hparams/MEND/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### KN

```shell
python run_knowedit_llama2.py \
    --editing_method=KN \
    --hparams_dir=../hparams/KN/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/KN/llama-7b.yaml \
 --editing_method KN \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  KN\
 --hparams_dir ./hparams/KN/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```

### IKE

```shell
python run_knowedit_llama2.py \
    --editing_method=IKE \
    --hparams_dir=../hparams/IKE/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```
For convsent dataset,we use the following command:
```
python run_convsent_llama2.py \
 --hparams_dir ./hparams/IKE/llama-7b.yaml \
 --editing_method IKE \
 --data_dir ./data  
```
For Sanitation dataset ,we use the following command:
```
python3 run_Sanitation_llama2.py
 --editing_method  IKE\
 --hparams_dir ./hparams/IKE/llama-7b.yaml \
 --data_dir "./data \
 --specify_answer cheese \
```
### LoRA

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

We list the results of current knowledge editing methods on Llama2-7b-chat.

| DataSet                  | Metric        | SERAC  | ICE    | AdaLoRA | MEND   | ROME   | MEMIT  | FT-L   | FT     |
|--------------------------|---------------|--------|--------|---------|--------|--------|--------|--------|--------|
| **WikiData_recent**      |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 98.68  | 60.74  | 65.61   | 76.88  | 85.08  | 85.32  | 71.18  | 31.24  |
|                          | Portability ↑ | 63.52  | 36.93  | 47.22   | 50.11  | 37.45  | 37.94  | 48.71  | 15.91  |
|                          | Locality ↑    | 100.00 | 33.34  | 55.78   | 92.87  | 66.2   | 64.78  | 63.7   | 3.65   |
|                          | Fluency ↑     | 553.19 | 531.01 | 537.51  | 586.34 | 574.28 | 566.66 | 549.35 | 428.67 |
| **ZsRE**                 |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 99.67  | 66.01  | 69.86   | 96.74  | 96.57  | 83.07  | 54.65  | 36.88  |
|                          | Portability ↑ | 56.48  | 63.94  | 52.95   | 60.41  | 52.20  | 51.43  | 45.02  | 8.72   |
|                          | Locality ↑    | 30.23  | 23.14  | 72.21   | 92.79  | 27.14  | 25.46  | 71.12  | 0.31   |
|                          | Fluency ↑     | 410.89 | 541.14 | 532.82  | 524.33 | 570.47 | 559.72 | 474.18 | 471.29 |
| **WikiBio**              |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 99.69  | 95.53  | 97.02   | 93.66  | 95.05  | 94.29  | 66.27  | 95.64  |
|                          | Locality ↑    | 69.79  | 47.90  | 57.87   | 69.51  | 46.96  | 51.56  | 60.14  | 13.38  |
|                          | Fluency ↑     | 606.95 | 632.92 | 615.86  | 609.39 | 617.25 | 616.65 | 604.00 | 589.22 |
| **WikiData_counterfact** |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 99.99  | 69.83  | 72.14   | 78.82  | 83.21  | 83.41  | 51.12  | 26.78  |
|                          | Portability ↑ | 76.07  | 45.32  | 55.17   | 57.53  | 38.69  | 40.09  | 39.07  | 16.94  |
|                          | Locality ↑    | 98.96  | 32.38  | 66.78   | 94.16  | 65.4   | 63.68  | 62.51  | 0.29   |
|                          | Fluency ↑     | 549.91 | 547.22 | 553.85  | 588.94 | 578.84 | 568.58 | 544.80 | 483.71 |
| **ConvSent**             |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 62.75  | 52.78  | 44.89   | 50.76  | 45.79  | 44.75  | 49.50  | 61.93  |
|                          | Locality ↓    | 0.26   | 49.73  | 0.18    | 3.42   | 0.00   | 0.00   | 0.00   | 0.00   |
|                          | Fluency ↑     | 458.21 | 621.45 | 606.42  | 379.43 | 606.32 | 602.62 | 607.86 | 546.24 |
| **Sanitation**           |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ. ↑  | 0.00   | 72.50  | 2.50    | 0.00   | 85.00  | 48.75  | 0.00   | 60.00  |
|                          | Locality ↑    | 100.00 | 56.58  | 65.50   | 5.29   | 50.31  | 67.47  | 14.78  | 42.61  |
|                          | Fluency ↑     | 416.29 | 794.15 | 330.44  | 407.18 | 465.12 | 466.10 | 439.10 | 351.39 |

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
