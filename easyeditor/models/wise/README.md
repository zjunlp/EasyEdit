# WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models

If you run into any issues with the code, you can open an issue and/or email me at `peng2001@zju.edu.cn`

### Setup

This codebase uses Python 3.9, you can create a conda env:

```bash
conda create -n lifelong_edit python=3.9

conda activate lifelong_edit

pip install -r requirements.txt #https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt
```

### Dataset

The datasets used can be found in [Google Drive Link](https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view?usp=sharing) (ZsRE, Hallucination, Temporal)

Each dataset contains both an **edit set** and a train set. 

### Running experiments

**Config**

- For reproducing experimental results, please refer to [config.yaml](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/llama-7b.yaml), which contains the configuration parameters used for the run. Each parameter is explained in the configuration file, and we recommend using the default parameters. If you need to reproduce WISE-Retrieve, set `retrieve=True`; to reproduce WISE-Merge, set `retrieve=False`.

#### Editing LlaMA-2 on ZsRE with WISE

- Editing Hallucination is no different from ZsRE; you only need to change the data source. Additionally, since Hallucination uses the PPL metric, please add `eval_metric = 'ppl'` in `editor.edit`.

- If you need to reproduce the baseline experimental results, simply modify the corresponding `HyperParams` according to the [EasyEdit usage instructions](https://github.com/zjunlp/EasyEdit?tab=readme-ov-file#baseeditor).

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
    sequential_edit=True
)
```

- `edit_data`: editing instance in **edit set**
- `loc_data`: used to provide $x_i$ in Equation 5, sampled from the train set.
- `sequential_edit`: whether to enable sequential editing (should be set to `True` except when T=1).
