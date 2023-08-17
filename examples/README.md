# Examples

We host a wide range of examples to elaborate the basic usage of EasyEdit. 

We also have some research projects that reproduce results in research papers using EasyEdit. 

Please discuss in an [issue](https://github.com/zjunlp/EasyEdit/issues) a feature you would  like to implement in an example before submitting a PR; we welcome bug fixes, but since we want to keep the examples as simple as possible it's unlikely that we will merge a pull request adding more functionality at the cost of readability.

---

## Data

The datasets used can be downloaded from [here](https://drive.google.com/file/d/1IVcf5ikpfKuuuYeedUGomH01i1zaWuI6). Unzip the file and put it to `./data`, and the final directory structure is as follows:

```text
editing-data
├── counterfact
│   ├── counterfact-original-edit.json
│   ├── counterfact-original-train.json
│   └── counterfact-original-val.json
├── locality
│   ├── Commonsense Task
│   │   ├── piqa_valid-labels.lst
│   │   └── piqa_valid.jsonl
│   ├── Distracting Neighbor
│   │   └── counterfact_distracting_neighbor.json
│   └── Other Attribution
│       └── counterfact_other_attribution.json
├── portability
│   ├── Inverse Relation
│   │   └── zsre_inverse_relation.json
│   ├── One Hop
│   │   ├── counterfact_portability_gpt4.json
│   │   └── zsre_mend_eval_portability_gpt4.json
│   └── Subject Replace
│       ├── counterfact_subject_replace.json
│       └── zsre_subject_replace.json
└── zsre
    ├── zsre_mend_eval.json
    └── zsre_mend_train_10000.json
```

- counterfact: original counterfact dataset using Entity replacement
- zsre: original question answering dataset using question rephrasings
- locality (evaluation for locality, see details in this [paper](https://arxiv.org/abs/2305.13172))
    - Commonsense Task: evaluation for other downstream tasks such as commonsense task
    - Distracting Neighbor: test on distracting neighborhood ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553))
    - Other Attribution
- portability
    - Inverse Relation: evaluation for one-to-one relationship such as `spouse`
    - One Hop: evaluation for one-hop reasoning
    - Subject Replace: evaluation for synonym replacement


## Edit llama-2 on ZsRE

In the paper [EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269), the data set we used is `zsre_mend_eval_portability_gpt4.json`, so you should place it in the `./data` directory.
```shell
python run_zsre_llama2.py \
    --editing_method=ROME \
    --hparams_dir=../hparams/ROME/llama-7b \
    --data_dir=./data
```
- `editing_method`: Knowledge Editing Method (e.g., `ROME`, `MEMIT`, `IKE`)
- `hparams_dir`: HyperParams Path.
- `data_dir`: dataset Path.

- Metric results for each editing are stored at `metrics_save_dir`(default: `./results.json`)



