<div align="center">
<h1> PROMPT-based editing methods </h1>
</div>

When humans encounter new information, we do not always master it immediately. Instead, with the right context and examples, we can process and reason through this new knowledge. LLMs exhibit a similar capacity for **in-context learning**. This README introduces two ICL-based knowledge editing methods currently supported by EasyEdit: one is `IKE`, and the other is a very simple `PROMPT` approach.

## 📚 Method Introduction
`IKE` (In-context Knowledge Editing), is a way of editing factual knowledge in large language models **without modifying their parameters**, it exemplifies this approach by constructing three types of demonstrations – **copy**, **update**, and **retain** – to aid the model in producing reliable fact editing. It utilizes a demonstration store, formed from training sets, to guide the model towards generating the appropriate answer by retrieving the most pertinent demonstrations.

<div align="center">
<img src="../figs/IKE.png" width="50%" height="50%" />
</div>

`PROMPT` approach is much simpler compared to the IKE method. It does not require specifying a training dataset, constructing three types of demonstrations, or involving a retrieval process. Instead, the fact to be edited in the data is directly treated as the **new fact**, and the model is prompted to response based on this **new fact**.

## 🚀 How to run
In EasyEdit, both the IKE and PROMPT methods share the same hyperparameter configuration file and execution function. When the `use_icl_examples` field in the hyperparameters is set to **False**, the PROMPT method is used. Conversely, when the `use_icl_examples` field is set to **True**, the IKE method is employed. By default, the `use_icl_examples` field is set to True.

### Run IKE

**hyperparameter**
```python
alg_name: "IKE"
model_name: "./hugging_cache/llama-3-8b"
sentence_model_name: "./hugging_cache/all-MiniLM-L6-v2"
device: 0
results_dir: "./results"
use_icl_examples: True

k: 16
model_parallel: false
```
**An example: Editing LlaMA-3 on KnowEdit with IKE**
```shell
python run_knowedit_llama2.py \
    --editing_method=IKE \
    --hparams_dir=./hparams/IKE/llama3-8b.yaml \
    --data_dir=./data/KnowEdit \
    --train_data_path='the_train_data_path'
```
- **train_data_path**: IKE needs `train_ds` (for getting In-Context prompts)

### Run PROMPT
**hyperparameter**
```python
alg_name: "IKE"
model_name: "./hugging_cache/llama-3-8b"
sentence_model_name: " "
device: 0
results_dir: "./results"
use_icl_examples: False

k: 0
model_parallel: false
```
- **sentence_model_name**: It can be directly set to `""`, because SentenceTransformer is no longer needed to assist in encoding sentences from the training dataset in this case.
- **K**: It can be directly set to `0`, because it is no longer necessary to add K ICL examples before the prompt in this case.

**An example: Editing LlaMA-3 on KnowEdit with PROMPT**
```shell
python run_knowedit_llama2.py \
    --editing_method=IKE \
    --hparams_dir=./hparams/IKE/llama3-8b.yaml \
    --data_dir=./data/KnowEdit \
    --train_data_path=' '
```
- **train_data_path**: Although the PROMPT method does not require a training dataset, the `train_data_path` field still needs to be explicitly specified.
