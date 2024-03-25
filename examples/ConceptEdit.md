<div align="center">

**Editing Conceptual Knowledge for Large Language Models**

![](https://img.shields.io/badge/version-v0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/PRs-Welcome-red)

---

<p align="center">
  <a href="#-conceptual-knowledge-editing">Overview</a> •
  <a href="#-usage">How To Use</a> •
    <a href="#-data-preparation">Data</a> •
    <a href="#-citation">Citation</a> •
    <a href="https://arxiv.org/abs/2403.06259">Paper</a> •
    <a href="https://zjunlp.github.io/project/ConceptEdit">Website</a> 
</p>
</div>


## 💡 Conceptual Knowledge Editing

<div align=center>
<img src="../figs/flow1.gif" width="70%" height="70%" />
</div>

### Task Definition

**Concept** is a generalization of the world in the process of cognition, which represents the shared features and essential characteristics of a class of entities.
Therefore, the endeavor of concept editing aims to modify the definition of concepts, thereby altering the behavior of LLMs when processing these concepts.


### Evaluation

To analyze conceptual knowledge modification, we adopt the  metrics for factual editing (the target is the concept $C$ rather than factual instance $t$).

- `Reliability`: the success rate of editing with a given editing description
- `Generalization`: the success rate of editing **within** the editing scope
- `Locality`: whether the model's output changes after editing for unrelated inputs


Concept Specific Evaluation Metrics

- `Instance Change`: capturing the intricacies of these instance-level changes
- `Concept Consistency`: the semantic similarity of generated concept definition


## 🌟 Usage

### 🎍 Current Implementation
As the main Table of our paper, four editing methods are supported for conceptual knowledge editing.
| **Method** |  GPT-2 | GPT-J | LlaMA2-13B-Chat | Mistral-7B-v0.1
| :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | 
| FT | ✅ | ✅ | ✅ | ✅ | 
| ROME | ✅ | ✅ |✅ | ✅ | 
| MEMIT | ✅ | ✅ | ✅| ✅ | 
| PROMPT | ✅ | ✅ | ✅ | ✅ |

> ❗️❗️ If you intend to use **"LlaMA2-13B-Chat"** rather than "LlaMA2-13B-Base", please modify the "model_name" in "./hparams/[METHOD]/llama-7b.yaml" or write the .yaml file by yourself.

### 🔧 Pip Installation

**Note: Please use Python 3.9+ for EasyEdit**

To get started, simply install conda and run:

```shell
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
...
conda activate EasyEdit
pip install -r requirements.txt
```

> ❗️❗️ If you intend to use Mistral, please update the `transformers` library to version 4.34.0 manually. You can use the following code: `pip install transformers==4.34.0`.

---


### 📂 Data Preparation

**Dataset for Conceptual Knowledge Editing: ConceptEdit** 
You can download it from [[Google Drive]](https://drive.google.com/drive/folders/1Hp1DfIuj6Ih6ZLVENS-UmgJT8mRBlFC2?usp=drive_link), then put the data in folder "./data".

**"concept_data.json"** is the main data file containing 452 concepts, 8,767 instances with 22 superclasses.

> ❗️❗️ For quick start, we preprocess the data for experiment on different settings and exhibit the post-processed files which are used in main Table. You can follow its format to build your file if needed.

<!-- **temp_stat for MEMIT**  -->

### 💻 Run

Before you begin running the program, ensure that the necessary files are present and properly set up, specifically the directories **./data, ./hparams,** and **./hugging_cache**. 

Also, move the file **run_concept_editing.py** to **./** (We will later modify the code to adapt to running in the current directory).

STEP 1 :
```shell
python run_concept_editing.py     --editing_method=ROME  --edited_model gptj   --hparams_dir=./hparams/ROME/gpt-j-6B  --inter
```

> Additional shell script examples for configuring experiments are available in the test_conceptEdit.sh file.


STEP 2 (OPTIONAL) :


Given that the generation task in LLMs can be time-consuming, if you wish to perform `Concept Consistency`, follow these instructions:

1. uncomment line 113 in the `run_concept_editing.py` file:   `concept_consistency = True`
2. Should you require generation of descriptions before editing, you need to modify line 184 in `easyeditor/editors/concept_editor.py`: `test_concept_consistency=concept_consistency`
3. With these adjustments, proceed to re-execute STEP 1.
4. To convert the generated sentences into a **Json** file for evaluating with GPT-4, execute the following command:
```shell
python examples/transform_check_concept.py --method ROME --model gptj --module inter
```


<!-- **Note:** Ensure these changes are done correctly to enable the 'Concept Consistency'  -->


## 📖 Citation

Please cite our paper if you use **ConceptEdit** in your work.

```bibtex
@misc{wang2024editing,
      title={Editing Conceptual Knowledge for Large Language Models}, 
      author={Xiaohan Wang and Shengyu Mao and Ningyu Zhang and Shumin Deng and Yunzhi Yao and Yue Shen and Lei Liang and Jinjie Gu and Huajun Chen},
      year={2024},
      eprint={2403.06259},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 🎉 Acknowledgement

We would like to express our sincere gratitude to [DBpedia](https://www.dbpedia.org/resources/ontology/)，[Wikidata](https://www.wikidata.org/wiki/Wikidata:Introduction)，[OntoProbe-PLMs](https://github.com/vickywu1022/OntoProbe-PLMs) and [ROME](https://github.com/kmeng01/rome).

Their contributions are invaluable to the advancement of our work.
