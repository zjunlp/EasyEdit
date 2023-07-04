<div align="center">

<img src="figs/logo.png" width="200px">

**An Easy-to-use Framework to Edit Large Language Models.**

---

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#use-easyinstruct">How To Use</a> •
  <a href="https://zjunlp.gitbook.io/easyinstruct/">Docs</a> •
  <a href="#citation">Citation</a> •
  <a href="#contributors">Contributors</a>
</p>
</div>

## 🌟Overview
EasyInstruct is a Python package for edit Large Language Models (LLM) like `GPT-J`, `Llama`, `GPT2`, `T5`, the objective of which is to alter the behavior of LLMs efficiently within a specific domain without negatively impacting performance across other inputs.  It is designed to be easy to use and easy to extend. The current supported model editing techniques:

- [FT-L](https://github.com/kmeng01/rome): Fine-Tuning with $L_\infty$ constraint
- [SERAC](https://github.com/eric-mitchell/serac): Mitchell et al. Memory-based
- [IKE](https://github.com/Zce1112zslx/IKE): Ce Zheng et al. In-Context Editing
- [KE](https://github.com/nicola-decao/KnowledgeEditor): De Cao et al. Knowledge Editor
- [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Hypernetwork
- [KN](https://github.com/Hunter-DDM/knowledge-neurons): Damai Dai et al. Locate then Edit
- [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
- [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit
---

## 🔧Installation
To get started, simply install conda and run:
```shell
conda create -n EasyEdit python=3.9.7
...
pip install -r requirements.txt
```
## 📌Use EasyEdit

### BaseEditor
> `BaseEditor` is the class for Language Modality Model Editing. You can choose the appropriate editing method based on your specific needs.

**Example**

```python
## Question prompt
prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]

## You can set `ground_truth` to None(original output)
ground_truth = [
    'Illinois Institute of Technology', 'Lecanorales', 'defender'
]

## Expected output
target_new = [
    'University of Michigan', 'Lamiinae', 'winger'
]
subject = [
    'Watts Humphrey', 'Ramalinaceae', 'Denny Herzig'
]
## Loading from hparams/MEMIT/gpt2-xl.yaml
hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl')

## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=True
)

## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model
```










