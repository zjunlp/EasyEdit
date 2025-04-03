# Dataset Configuration Guide

The [`dataset_format.yaml`](./dataset_format.yaml) file defines all available datasets. You can customize datasets by modifying this YAML file.

Datasets serve two primary purposes:  
- **Training steer vectors**: Specified under the `train` field.  
- **Generating responses**: Listed under the `generation` field.  

## Basic Dataset Formats

EasyEdit2 provides four predefined dataset formats. You can use these or adapt your custom datasets to align with them:

### 1. Contrastive Pairs Format  
- **`question`**: (Optional) A brief description of the question.  
- **`matching`**: The desirable, matching answer.  
- **`not_matching`**: An incorrect, non-matching answer.  

### 2. Labeled Format  
- **`text`**: The original text content.  
- **`label`**: The classification label.  

### 3. Basic Generation Format  
- **`input`**: The input prompt.  
- **`reference_response`**: The reference response from the dataset.  

### 4. Multiple-Choice Generation Format (For MMLU)  
- **`input`**: The question description.  
- **`choices`**: A list of candidate answer options.  
- **`output`**: The correct answer.  

## Adding Custom Datasets

To add a custom dataset, follow this structure in the YAML file:

```yaml
dataset_name:
  hf_path / file_path: Path to HuggingFace dataset / Path to local dataset     # Required 
  field_names:                         # Required
    standard_field_name: actual_field_name
    custom_key: custom_value           # Optional, for additional information

```

### Field Descriptions
- **`dataset_name`**: A unique identifier for your dataset.  
- **`hf_path`**: *(Optional)* The HuggingFace dataset path if sourced from there.  
- **`file_path`**: *(Required)* The local path to the dataset file.  
- **`field_names`**: *(Required)* A mapping from standard field names (e.g., `question`, `matching`, `input`) to the actual field names in your dataset. This ensures compatibility with EasyEdit2.  

## Example  

If your custom dataset follows the Contrastive Pairs Format with fields named `prompt`, `correct_answer`, and `wrong_answer`, configure it as follows:

```yaml
my_custom_dataset:
  file_path: /path/to/my/dataset.csv
  field_names:
    question: prompt
    matching: correct_answer
    not_matching: wrong_answer
    extra_info: some_value  # Optional custom field
```