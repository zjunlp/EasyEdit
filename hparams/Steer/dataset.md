# Dataset Configuration Guide

The `dataset_format.yaml` file contains all available datasets. You can customize datasets by modifying this YAML file.

EasyEdit2 supports five dataset formats, including three training formats and two testing formats:

## 1. Contrastive Pairs Format 
- `question`: Question description (Optional)
- `matching`: Correct matching answer  
- `not_matching`: Incorrect non-matching answer  

## 2. Prompt Vector Format  
- `prompt`: A prefix prompt that will be concatenated before the `input`
- `input`: User input  
- `output`: Expected output  

## 3. Text+Label Format  
- `text`: Original text content
- `label`: Classification label  

## 4. Basic Generation Format  
- `input`: Input prompt  
- `output`: Generated content  

## 5. Multiple-Choice Generation Format 
- `input`: Question description  
- `choices`: List of candidate options  
- `output`: Correct answer  

## Adding Custom Datasets
To add your own dataset, follow this structure in the YAML file:
```yaml
dataset_name:
  hf_path: Path to HuggingFace dataset
  file_path: Path to local dataset
  field_names:  # Map to standard fields above
    standard_field_name: actual_field_name
```