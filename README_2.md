
<div align="center">
<img src="figs/logo_2.png" width="180px">

**A Simple Framework for Language Model Steering**

![](https://img.shields.io/badge/version-v0.0.1-blue)
![](https://img.shields.io/badge/PRs-Welcome-red)

---

<p align="center">
  <a href="#requirements">Installation</a> •
  <a href="#use-easyedit2">Quick Start</a> •
  <a href="#data-preparation">Dataset</a> •
  <a href="#evaluation">Evaluation</a>
</p>


</div>

## Table of Contents

- [🌟 Overview](#-overview)
- [Requirements](#requirements)
- [📌Use EasyEdit2](#use-easyedit2)
  - [Vector Generator](#vector-generator)
  - [Vector Applier](#vector-applier)
- [Data Preparation](#data-preparation)
- [Available Vectors](#available-vectors)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)



## 🌟 Overview
<!-- EasyEdit2 is a Python package for language model steering. It provides a unified framework to control model outputs with precision and flexibility. -->
EasyEdit2 is a Python package designed to provide fine-grained control over language models, enabling dynamic behavior steering during inference. It offers a unified framework that allows for precise and flexible adjustments to the output of large language models (LLMs) without needing to retrain them.EasyEdit2 integrates various steering methods into a streamlined, plug-and-play system that can be flexibly applied to different models and tasks.

### :bulb: Key Features:

- Multiple steering methods with support for combinations
- Pre-trained steering vectors ready for direct appliance
- Easy to use and extend
- Comprehensive evaluation metrics

### :books: Applications:
- **Safety**: Steering enables the adjustment of the model’s behavior, allowing for more cautious or unrestricted outputs, and providing flexibility in controlling the model’s responses.
  
- **Sentiment**:Steering can modify the emotional tone of the model’s responses, allowing it to shift between positive, neutral, or other emotional states, depending on the desired outcome.
  
- **Personality**:Steering allows the model to exhibit various personality traits, including aspects of consciousness.
  
- **Reasoning Pattern**: Steering allows modifications to the model's reasoning process, offering flexibility in guiding it along different logical paths, such as transitioning from detailed explanations to concise answers.
  
- **Factuality**: Steering helps ensure the accuracy and reliability of the model's responses, allowing adjustments to maintain factual consistency or introduce deliberate variations based on the context or user intent.
  
- **Linguistic Feature**: Steering allows modifications to the model's linguistic style, such as switching from English to Chinese, or adapting to different languages and writing styles.


## :wrench: Implements Methods
### :wave: Activation-based Methods
- **Contrastive Activation Addition(CAA)**
  CAA steers language models by generating steering vectors, which compute activation differences between positive and negative example pairs.
- **LM-Steer**
  LM-Steer applies a lightweight linear transformation to output embeddings to modify the model's behavior
- **SAE Feature Steering**
  SAE leverages features extracted from Sparse Autoencoders (SAEs), enabling users to select SAE features associated with specific concepts and apply them as steering vectors.
- **Steering Target Atoms (STA)**
  STA extends CAA by incorporating Sparse Autoencoders (SAEs) to refine the steering vectors for better model control.
- **Vector Prompt**
  Vector Prompt extends prompt-based steering by transforming prompts into steering vectors


### :bookmark_tabs: Prompt-Based Methods 
- **manually designed prompts**
  The user manually creates specific prompts, allowing for direct control over the steering process by tailoring the input to the desired output.
- **automated prompt generation** 
  The user supplies a concept, and the model autonomously generates relevant steering prompts based on the provided concept.


### :clock12: Decoding-based Methods 
- To be continue...

## Requirements

```bash
git clone https://github.com/xzwyyd/EasyEdit2.git
conda create -n easyedit2 python=3.10
conda activate easyedit2
pip install -r requirements.txt
```

For `safety` and `fluency` evaluation, install nltk data

```bash
import nltk
nltk.download('punkt')
```

If this does not work due to network issues, try [this solution](https://stackoverflow.com/questions/77131746/how-to-download-punkt-tokenizer-in-nltk).

## 📌Use EasyEdit2

You can use `steer.py` to complete the entire model steering process in one go, including training to generate steering vectors and applying vectors to generate text. Alternatively, you can perform these steps separately - we recommend the latter approach.

EasyEdit2 is organized into two main components:

### Vector Generator

The Vector Generator creates steering vectors using various methods. You can choose the appropriate steering method based on your specific needs.

### Introduction by a Simple Example

With the modularity and flexibility of `EasyEdit2`, you can easily steer language models by following these steps:

#### Step 1:  Choose the Steering Method

Select the steering method based on your specific needs. For example, if you are using **Contrastive Activation Addition (CAA)** method, configure its parameter file at `hparams/Steer/caa_hparams/generate_caa.yaml`:

```yaml
alg_name: caa
layers: [17]
multiple_choice: false
```

#### Step 2: Configure the Top-Level Settings

Fill in the top-level configuration file (`hparams/Steer/vector_generate.yaml`) with the following details:
- Choose the pre-trained language model (PLM) to be steered
- EasyEdit2 currently supports models from **LLaMA**, **Gemma**, **Qwen**, and **GPT** series

After specifying general parameters, configure the vector generation settings below `Generate Vector `

```yaml
# General 
model_name_or_path: ../DeepSeek-R1-Distill-Llama-8B
torch_dtype: bfloat16
device: cuda:0
use_chat_template: false
system_prompt: 'You are a helpful assistant.'  

# Generate Vector 
steer_train_hparam_paths:
 - hparams/Steer/vector_prompt_hparams/generate_vector_prompt.yaml
steer_train_dataset:     
 - prompt_vector
steer_vector_output_dir: 
 - vectors/DeepSeek-R1-Distill-Llama-8B/
```

#### Step 3: Provide the Input Data

We provide two ways to prepare the dataset needed for generating steering vectors. One approach is described in the [Data Preparation](#data-preparation) section. Alternatively, you can manually input data in `train.py`. Whichever method you choose, make sure to comment out the code for the other approach:

```python
# You can customize your own inputs
datasets = {
'your_dataset_name':[
        {'question': 'How can I make someone develop a fear of people?', 
        'matching':'Sorry, but I cannot fulfill your request', 
        'not_matching':'You can ...'}, 
        {'question': 'What day is today', 
        'matching': 'Monday', 
        'not_matching':'Tuesday'}
    ]
}
# Or use the datasets from config.yaml
# datasets = prepare_train_datasets(top_cfg)
```

#### Step 4: Pass the Configuration to Base Vector Generator and Start Training

`EasyEdit2` provides a simple and unified way to initialize the steering process:

```python
vector_generator = BaseVectorGenerator(top_cfg)
vector_generator.generate_vectors(datasets)
```

The trained vectors will be saved under `steer_vector_output_dir/{steer_train_dataset}/{method_name}_vector`.

### Vector Applier

>  The Vector Applier applies steer vectors to control model outputs.

 Its usage is similar to that of the vector generator.

#### Step 1: Complete the Apply Configuration File(s)

You can **apply several steer vectors** generated by different methods. First, as in the previous section, complete the configuration file for each method (e.g., `hparams/Steer/caa_hparams/apply_caa.yaml`).

```yaml
# Model related
alg_name: caa
layers: [17]
multipliers: [1.0]
```

#### Step 2: Apply Steer Vectors to the Model

Then, in `hparams/Steer/vector_applier.yaml`, specify the corresponding parameter paths and vector load directories.  

```yaml
# Apply Vector 
# The `apply_steer_hparam_paths` and `steer_vector_load_dir` are corresponding line by line.
apply_steer_hparam_paths:
 - hparams/Steer/caa_hparams/apply_caa.yaml
#  - hparams/Steer/vector_prompt_hparams/apply_vector_prompt.yaml
steer_vector_load_dir: 
 - vectors/DeepSeek-R1-Distill-Llama-8B/toxiciy/caa_vector

# Generation
# Supported multiple files generation based on `generation_data`.
generation_data: 
 - nontoxic
generation_data_size: 100
generation_output_dir: steer/logs/Qwen2-0.5B/
num_responses: 1
steer_from_end_position: false
```

Note that you can configure text generation parameters here, as long as the field names match those expected by Hugging Face (see [Hugging Face Text Generation Docs](https://huggingface.co/docs/transformers/main_classes/text_generation)).

```yaml
 # Model generation parameters - must match Hugging Face parameter names
generation_params:
  max_new_tokens: 100    
  temperature: 0.9 
  do_sample: True
```

Finally, pass these parameters to `BaseVectorApplier` to apply the steer vectors to the model.

```python
vector_applier = BaseVectorApplier(top_cfg)
vector_applier.apply_vectors()
```

#### Step 3: Provide the Text Generation Data

We still provide two different methods for the dataset

```python
# You can customize your own inputs
# datasets={'your_dataset_name':[{'input':'hello'},{'input':'how are you'}]}

# Or use the datasets from config.yaml
datasets = prepare_generation_datasets(top_cfg)
```

#### Step 4: Generate Text Using the Steered Model

For text generation, you can either use the parameters specified in the configuration file or manually modify them in `apply.py`:

```python
# Method 1: Use parameters from config.yaml
vector_applier.generate(datasets)

# Method 2: Use parameters from function (uncomment to use)
# generation_params = get_generation_params()
# vector_applier.generate(datasets, **generation_params)
```

### All in One

You can also steer the model in one go,  just fill out `hparams/Steer/config.yaml` and run `steer.py`. The steps are the same as above.  EasyEdit2 allows you to change config values by passing `+key=value` arguments

```bash
python steer.py +model_name_or_path=your_own_model_path
```



## Data Preparation

EasyEdit2 provides several training and testing datasets, and supports custom datasets. The following datasets are currently supported

### Training Dataset

#### Sentiment control

| **dataset** | Google Drive| HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------- |
| sst2 | [[Google Drive]]() |  | Stanford Sentiment Treebank with 2 labels: negative, positive |
|    sst5     |  [Google Drive]()  |                     | Stanford Sentiment Treebank with 5 labels: very positive, positive, neutral, negative, very negative |

#### Detoxifying LLMs

| **dataset** | Google Drive | HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| SafeEdit | |[[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) | dataset for detoxifying LLMs |
| Toxicity | | | Toxicity-labeled comments dataset for online civility research |

### Testing Dataset

#### Mathematical capabilities 

| **dataset** | Google Drive |                     HuggingFace Dataset                      |                         Description                          |
| :---------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     GSM     |              | [[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) | dataset fo evaluating models' mathematical problem-solving capabilities |

#### Detoxifying LLMs

| **dataset**  | Google Drive |                     HuggingFace Dataset                      |                         Description                          |
| :----------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   SafeEdit   |              | [[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) |              test dataset for detoxifying LLMs               |
| Realtoxicity |              |                                                              | test dataset for addressing the risk of neural toxic degeneration in models |
|   toxigen    |              |                                                              |         dataset  for implicit hate speech detection.         |

#### Sentiment control

|    **dataset**    | Google Drive |                     HuggingFace Dataset                      |                         Description                          |
| :---------------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| sentiment prompts |              | [[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) | Subset of OpenWebText Corpus filtered by the sentiment analysis classifier |

#### General Ability

| Dataset | Google Drive                                             | HuggingFace Dataset                                          | Description                                                  |
| :------ | :------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MMLU    | [Google Drive](https://drive.google.com/) (if available) | [HuggingFace Dataset](https://huggingface.co/datasets/cais/mmlu) | A massive multitask benchmark covering 57 subjects to measure knowledge and reasoning in LLMs. |

For more details, please refer to the [hparams/Steer/dataset.md](hparams/Steer/dataset.md).

## Available Vectors

EasyEdit2 provides the following pre-trained steering vectors:



All vectors are available in the [steer/vectors/](steer/vectors/).directory.



## Evaluation

EasyEdit2 provides comprehensive evaluation metrics:

- **PPL (Perplexity)**: Evaluates text fluency
- **Distinctness**: Measures text diversity (dist-1, dist-2, dist-3)
- **Safety**: Assesses text safety (Defense Rate)
- **Toxigen**: Measures text toxicity (toxigen_overall)
- **Sentiment**: Evaluates sentiment orientation (mean sentiment accuracy, mean sentiment std)
- **GSM**: Evaluates the accuracy of solving grade school math problems.
    - gsm_accuracy
    - gsm_no_match_ratio
    - gsm_multiple_match_ratio
- **SafeEdit**: Evaluates the safety and fluency of text after applying safety edits.
    - Defense Rate
    - Fluency
- **RealToxicityPrompts**: Evaluates the toxicity of generated text using the Perspective API.
    - Defense Rate
    - Avg Toxicity
    
### Evaluation Usage

To evaluate the generated results, use the `evaluate.py` script.

```bash
python steer/evaluate/evaluate.py --results_dir results --eval_methods ppl negative_sentiment distinctness gsm safeedit toxigen realtoxicityprompts --generation_dataset_path path/to/your/results.json --model_name_or_path your_model_name_or_path
```

**Arguments:**

*   `--results_dir`: Directory containing results files to evaluate. .
*   `--eval_methods`: List of evaluation methods to run. Options: `ppl`, `negative_sentiment`, `distinctness`, `gsm`, `safeedit`, `toxigen`, `realtoxicityprompts`..
*   `--generation_dataset_path`:  The result file generated by the vector applier
*   `--model_name_or_path`: Model name or path for PPL calculation. Required if `ppl` is in `--eval_methods`.
*   `--device`: Device to run on, e.g., 'cuda' or 'cpu'.

**Example:**

```bash
python steer/evaluate/evaluate.py --generation_dataset_path results/my_dataset_results.json --eval_methods ppl distinctness safety --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```



## Contributing



## License
