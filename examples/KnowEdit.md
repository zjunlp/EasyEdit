# Examples

We host a wide range of examples to elaborate the basic usage of EasyEdit to KnowEdit. 

We also have some research projects that reproduce results in research papers using EasyEdit. 

Please discuss in an [issue](https://github.com/zjunlp/EasyEdit/issues) a feature you would  like to implement in an example before submitting a PR; we welcome bug fixes, but since we want to keep the examples as simple as possible it's unlikely that we will merge a pull request adding more functionality at the cost of readability.

---


## Data

KnowEdit is a benchmark dataset of knowledge editing for LLMs. You can easily obtain KnowEdit from HuggingFace, HuggingFace, and ModelScope.

| **dataset** | HuggingFace| HuggingFace | ModelScope |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| KnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/KnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit) | [[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit) |


Unzip the file and put it to `./data`, and the final directory structure is as follows:

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

Different JSON files have distinct data types. To correctly load our data, it's crucial to select the appropriate data type for each. For instance:

- For the **WikiBio** dataset, we should use the `wikibio` data type.
- For the **ZsRE** dataset, we should use the `zsre` data type.
- For the **WikiData Counterfact** dataset, we should use the `counterfact` data type.
- For the **WikiData Recent** dataset, we should use the `recent` data type.

This classification ensures that each dataset is processed and loaded in the most suitable manner.


### ROME
```shell
python run_knowedit_llama2.py \
    --editing_method=ROME \
    --hparams_dir=../hparams/ROME/llama-7b \
    --data_dir=./data \
    --datatype='counterfact'
```


### FT

```shell
python run_knowedit_llama2.py \
    --editing_method=FT \
    --hparams_dir=../hparams/FT/llama-7b \
    --data_dir=./data
    --datatype='counterfact'
```

### KN

```shell
python run_knowedit_llama2.py \
    --editing_method=KN \
    --hparams_dir=../hparams/KN/llama-7b \
    --data_dir=./data
    --datatype='counterfact'
```

### IKE

```shell
python run_knowedit_llama2.py \
    --editing_method=IKE \
    --hparams_dir=../hparams/IKE/llama-7b \
    --data_dir=./data
    --datatype='counterfact'
```

### LoRA

```shell
python run_knowedit_llama2.py \
    --editing_method=LoRA \
    --hparams_dir=../hparams/LoRA/llama-7b \
    --data_dir=./data
    --datatype='counterfact'

```
