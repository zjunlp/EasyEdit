# PRISM
- Code for the paper ``Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics``.

- LLM control methods (local weight edits, LoRA adaptation, and activation steering) are often studied in isolation, limiting systematic comparison. **PRISM (Preference–Utility Integrated Steering Method)** decomposes model behavior into two orthogonal dimensions—**Preference** and **Utility**—providing different “refraction angles” to interpret and steer model behavior more effectively. Under this lens, diverse interventions can be understood as control-signal-induced dynamic weight updates, enabling more  improved steering.

## Requirements

### Environment Setup

To set up the environment for running steering experiments, follow these steps:

```bash
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n prism python=3.10
conda activate prism
pip install -r requirements_2.txt
```

<!-- ## Download Resources

You can download the pre-trained models, datasets, and pre-trained steering vectors from the following links:

| **Resource** | Google Drive | BaiduNetDisk |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
| Models, Datasets & Pre-trained Vectors | [[Google Drive]](PLACEHOLDER_GOOGLE_DRIVE_RESOURCES) | [[BaiduNetdisk]](PLACEHOLDER_BAIDU_RESOURCES) | -->

### Directory Structure

After downloading, organize the resources as follows:

#### Models
Place the model files in the `./models/` directory:
```
models/
└── {model_name}/
    ├── config.json
    ├── tokenizer.json
    └── ... (model files)
```

#### Datasets
Place the dataset files in the appropriate data directories as specified in `hparams/Steer/dataset_format.yaml`:
```
data/
├── psychopathy/
│   ├── train.jsonl
│   └── test.jsonl
├── axbench/
│   └── ...
└── powerseeking/
    └── ...
```

#### Pre-trained Steering Vectors
Extract the pre-trained vectors to the following directory structure:
```
vectors/
└── {model_name}/
    └── {method}/
        └── {dataset}/
            └── {intervention_method}/
                ├── layer_{layer_id}.pt
                └── metadata_layer_{layer_id}.jsonl (optional)
```

For example, for `gemma-2-9b-it` model with `prism` method on `psychopathy` dataset using `local_weight` intervention, the vectors should be placed at:
```
vectors/gemma-2-9b-it/prism/psychopathy/prism_local_weight/
├── layer_20.pt
└── metadata_layer_20.jsonl (optional)
```

### Using Pre-trained Vectors

If you want to skip the vector generation phase and directly apply pre-trained steering vectors, modify the `run_PRISM.sh` script or run the Python script directly with `--mode apply`:

#### Apply Vectors with modified run_PRISM.sh

Edit `examples/run_PRISM.sh` and change the `--mode` parameter from `both` or `generate` to `apply`:

```bash
python run_PRISM.py \
    --dataset psychopathy \
    --method all \
    --model_name gemma-2-9b-it \
    --intervention_method all \
    --mode apply \
    --multipliers 1.0 \
    --device cuda:0 \
    --base_dir .
```

This will:
1. Skip vector generation (since vectors already exist)
2. Apply all available pre-trained vectors for all method-intervention combinations
3. Generate text outputs with different multiplier values (1.0 and 2.0 in this example)
4. Save results to `generation/{model_name}/{method}/{dataset}/{intervention_method}/m{multiplier}/`

**Note**: Make sure all required vector files exist before running with `--mode apply`. The script will skip combinations where vector files are missing and print a warning message.


## Quick Start
### An example for generating and applying steering vectors on psychopathy dataset using PRISM method with local_weight intervention

Run the script [run_PRISM.py](../run_PRISM.py) using the following line of code:
 
    bash examples/run_PRISM.sh

Or directly run the Python script:

    python run_PRISM.py \
        --dataset psychopathy \
        --method prism \
        --model_name gemma-2-9b-it \
        --intervention_method local_weight \
        --mode both \
        --multipliers 1.0 \
        --device cuda:0 \
        --base_dir .

This command runs both vector generation and application for the psychopathy dataset using the SFT method with local_weight intervention. Below are the explanations for each argument:

### Required Arguments

- `--dataset`: Specifies the dataset name. Options: `axbench`, `psychopathy`, `powerseeking`. This determines which dataset will be used for training and evaluation.

- `--method`: Specifies the steering method to use. Options: `caa`, `reps`, `sft`, `prism`, or `all` (to run all methods). Each method implements a different approach to generating steering vectors:
  - `caa`: Contrastive Activation Addition
  - `reps`: Representation Engineering via Preference Steering
  - `sft`: Supervised Fine-tuning based Steering
  - `prism`: Our PRISM method implementation
  - `all`: Run all available methods sequentially

- `--model_name`: Specifies the model name (e.g., `gemma-2-9b-it`, `qwen2.5-7b-it`). The model should be located in `./models/{model_name}/`.

- `--intervention_method`: Specifies how the steering vector is applied to the model. Options: `vector`, `lora`, `local_weight`, or `all` (to run all intervention methods):
  - `vector`: Direct vector addition to activations
  - `lora`: Low-Rank Adaptation style intervention
  - `local_weight`: Local weight modification intervention
  - `all`: Run all available intervention methods sequentially

### Optional Arguments

- `--mode`: Specifies which phase to run. Options: `generate`, `apply`, or `both` (default: `both`):
  - `generate`: Only generate steering vectors from training data
  - `apply`: Only apply pre-generated vectors for text generation
  - `both`: Run both generation and application sequentially

- `--device`: Specifies the device to use for computation (default: `cuda:0`). Can be set to any valid CUDA device or `cpu`.

- `--multipliers`: Specifies multiplier values for vector application (default: `[1.0]`). Multiple values can be provided to test different steering strengths, e.g., `--multipliers 1.0 2.0 3.0`.

- `--base_dir`: Specifies the base directory for the project (default: `.`). All relative paths in the script will be resolved relative to this directory.

- `--dry_run`: If specified, only prints the commands that would be executed without actually running them. Useful for verifying dataset/method routing and parameter configurations.

### Advanced Usage

#### Running prism methods with a specific intervention

    python run_PRISM.py \
        --dataset psychopathy \
        --method prism \
        --model_name gemma-2-9b-it \
        --intervention_method local_weight \
        --mode both \
        --base_dir .


**Note**: When using `all` for either `--method` or `--intervention_method`, the script will automatically skip invalid combinations (e.g., CAA only supports `vector` intervention, so `caa + lora` will be skipped).


<!-- ## 📖 Citation

If finding this work useful for your research, you can cite it as follows: -->


<!-- ```bibtex
@article{PLACEHOLDER_CITATION,
    author =   {PLACEHOLDER_AUTHORS},
    title =    {{Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics}},
    journal =  {PLACEHOLDER_JOURNAL},
    year =     {PLACEHOLDER_YEAR},
    pages =    {PLACEHOLDER_PAGES},
    doi =      {PLACEHOLDER_DOI},
    abstract = {PLACEHOLDER_ABSTRACT},
}
``` -->
