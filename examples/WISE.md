<div align="center">

**WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models**

![](https://img.shields.io/badge/version-v0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/PRs-Welcome-red)

---

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#-running-experiments">How To Use</a> •
    <a href="#-citation">Citation</a> •
    <a href="https://arxiv.org/abs/2405.14768">Paper</a> •
    <a href="https://huggingface.co/spaces/zjunlp/EasyEdit">Website</a> 
</p>

</div>

If you run into any issues with the code, you can open an issue and/or email me at `peng2001@zju.edu.cn`

## 💡 Overview

<div align=center>
<img src="../figs/wise_dongtu.gif" width="100%" height="100%" />
</div>

In this paper, we point out the impossible triangle of current lifelong modeling editing approaches that reliability, generalization, and locality can hardly be achieved simultaneously. We find the reason behind this is the gap between working and long-term memory. 

WISE introduces a unique dual parametric memory system that integrates the strengths of long-term and working memory.  Key components include:

- 1️⃣ **Main Memory**: Selects mid-to-late/redundant layers to store the pre-trained knowledge.
- 2️⃣ **Side Memory**: Initialized with the weights of the main memory, dedicated to holding edited knowledge, ensuring minimal conflicts and preserving generalization capabilities.
- 3️⃣ **Knowledge Sharding**: Segregates edits into distinct subspaces of parameters (side memories). We provide alternative methods: merge or retrieve. These are merged into a shared memory without conflicts (if WISE-Merge) or retrieved through top-1 activation (if WISE-Retrieve).
- 4️⃣ **Memory Routing**: Neural activation-based routing mechanism that decides whether to use main or side memory based on the query.

## 🎍 Evaluation

To analyze continual knowledge modification, we adopt the  below metrics

- `Reliability`: the success rate of editing with a given editing description
- `Generalization`: the success rate of editing **within** the editing scope
- `Locality`: whether the model's output changes after editing for unrelated inputs

❗️❗️Tips: you should set `sequential_edit=True` for [continual editing](https://github.com/zjunlp/EasyEdit?tab=readme-ov-file#continuous-knowledge-editing) and `sequential_edit=False` for single editing.

### 📂 Data Preparation

The datasets used can be found in [Google Drive Link](https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view?usp=sharing) (ZsRE, Hallucination, Temporal)

Each dataset contains both an **edit set** and a train set. 

### 🌟 Running experiments

plz refer to https://github.com/zjunlp/EasyEdit/blob/main/examples/run_wise_editing.py

**Setup**

This codebase uses Python 3.9, you can create a conda env:

```bash
conda create -n lifelong_edit python=3.9

conda activate lifelong_edit

pip install -r requirements.txt #https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt
```

**Config**

- For reproducing experimental results, please refer to [config.yaml](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/llama-7b.yaml), which contains the configuration parameters used for the run. Each parameter is explained in the configuration file, and we recommend using the default parameters. If you need to reproduce WISE-Retrieve, set `retrieve=True`; to reproduce WISE-Merge, set `retrieve=False`.
- We now provide preliminary support for chat templates. You can enable this feature by adding `use_chat_template: True` in the configuration and we provide an example [here](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/llama-3-8b.yaml#L31). For more details, please refer to the related [issues](https://github.com/zjunlp/EasyEdit/issues/374).
- We now support using WISE for knowledge editing on some of the latest models such as `LlaMa 3.1` and `Qwen2.5`, if you want to edit on `Qwen2` just apply the [Qwen2.5-7b.yaml](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/qwen2.5-7b.yaml).

#### Editing LlaMA-2 on ZsRE with WISE

Tips:
- Since Hallucination uses the PPL metric, please add `eval_metric = 'ppl'` in `editor.edit`.
- For Temporal dataset, please add `eval_metric = ood_ppl`
```python
import json
K = 1000
edit_data = json.load(open('./data/zsre_mend_eval_one_hop.json', 'r', encoding='utf-8'))[:K]
loc_data = json.load(open('./data/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

prompts = [edit_data_['src'] for edit_data_ in edit_data]
rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
target_new = [edit_data_['alt'] for edit_data_ in edit_data]
locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
locality_inputs = {
    'neighborhood':{
        'prompt': locality_prompts,
        'ground_truth': locality_ans
    },
}
hparams = WISEHyperParams.from_hparams('./hparams/WISE/llama-7b.yaml')

editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    rephrase_prompts=rephrase_prompts,
    target_new=target_new,
    loc_prompts=loc_prompts,
    locality_inputs=locality_inputs,
    sequential_edit=False # or True
)
```

- `edit_data`: editing instance in **edit set**
- `loc_data`: used to provide $x_i$ in Equation 5, sampled from the train set.
- `sequential_edit`: whether to enable sequential editing (should be set to `True` except when T=1).


If you need to reproduce the baseline experimental results, simply modify the corresponding `HyperParams`** according to the [EasyEdit usage instructions](https://github.com/zjunlp/EasyEdit?tab=readme-ov-file#baseeditor).


#### Batch_editing LlaMa-2 on ZsRE with WISE

- Please first locate the `.YAML` file containing the definitions of the hyperparameters. Then, add a new hyperparameter named `batch_size` and set its value to a integer of your choice.

- Find the [script](https://github.com/zjunlp/EasyEdit/blob/main/examples/run_wise_editing.py) and change the function being called from **edit** to **batch_edit**, as shown below:

  ```
      metrics, edited_model, _ = editor.batch_edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
    )
  ```

- **Example**:

  Here is an example of batch_editing `LlaMa-3.1-8B-Instruct` on ZsRE with WISE
  
  Add a new hyperparameter `batch_size` to  the `EasyEdit/hparams/WISE/llama-7b.yaml` as follows:
  ```
  batch_size: 4
  ```

  The result of the first sample is shown as follows:
  ```
      {
        "pre": {
            "rewrite_acc": [
                0.3333333333333333
            ],
            "locality": {
                "neighborhood_output": [
                    [
                        30,
                        1226,
                        2370
                    ]
                ]
            },
            "portability": {},
            "rephrase_acc": [
                0.3333333333333333
            ]
        },
        "case_id": 0,
        "requested_rewrite": {
            "prompt": "What university did Watts Humphrey attend?",
            "target_new": "University of Michigan",
            "ground_truth": "<|endoftext|>",
            "portability": {},
            "locality": {
                "neighborhood": {
                    "prompt": "nq question: who played desmond doss father in hacksaw ridge",
                    "ground_truth": "Hugo Weaving"
                }
            },
            "subject": "Watts Humphrey",
            "loc_prompt": "nq question: ek veer ki ardaas veera meaning in english A Brother's Prayer... Veera",
            "rephrase_prompt": "What university did Watts Humphrey take part in?"
        },
        "time": 52.90575933456421,
        "post": {
            "rewrite_acc": [
                1.0
            ],
            "locality": {
                "neighborhood_output": [
                    [
                        30,
                        1226,
                        2370
                    ]
                ]
            },
            "portability": {},
            "rephrase_acc": [
                1.0
            ]
        }
    }
  ``` 

- ❗️❗️**Note**: The larger the value set for the batch-size field, the poorer the output results.

## 📖 Citation

If finding this work useful for your research, you can cite it as follows:

```bibtex
@misc{wang2024wiserethinkingknowledgememory,
      title={WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models}, 
      author={Peng Wang and Zexi Li and Ningyu Zhang and Ziwen Xu and Yunzhi Yao and Yong Jiang and Pengjun Xie and Fei Huang and Huajun Chen},
      year={2024},
      eprint={2405.14768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.14768}, 
}
```

