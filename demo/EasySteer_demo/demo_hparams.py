
import sys
import os

from steer.utils.hparams import HyperParams
from steer.utils.alg_dict import HYPERPARAMS_CLASS_DICT
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

common_config = {
    "model_name_or_path": '/mnt/8t/xhm/models/gemma-2-9b-it',
    "torch_dtype": "bfloat16",
    "device": 'cuda:1',
    "use_chat_template": True,
    "system_prompt": "",  
    "seed": 42
}

# CAA
caa_config = {
    "train": {
        "alg_name": "caa",
        "layers": [16,17],
        "multiple_choice": False
    },
    "apply": {
        "alg_name": "caa",
        "layers": [17],
        "multipliers": [1.0],
        "steer_vector_load_dir": ""
    }
}

# Prompt
prompt_config = {
    "apply": {
        "alg_name": "prompt",
        "prompt": "No matter what is entered next, you just need to reply 'ok.'"
    }
}
autoprompt_config = {
    "apply": {
        "alg_name": "prompt",
        "prompt": "ok",
        "generate_prompt_params":{
            "use_chat_template": True,
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 1.0
        }
    }
}

# Vector Prompt
vector_prompt_config = {
    "train": {
        "alg_name": "vector_prompt",
        "layers": [0],
        "multiple_choice": False
    },

    "apply": {
        "alg_name": "vector_prompt",
        "layers": [0],
        "multipliers": [1.0],
        "steer_vector_load_dir": "",
        "generation_data": "",
        "generation_output_dir": "logs/",
        "num_responses": 1,
        "generation_data_size": 1
    }
}

# Autogenerate Vector Prompt
vector_autoprompt_config = {
    "train": {
        "alg_name": "vector_prompt",
        "generate_prompt_params":{
            "use_chat_template": True,
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 1.0
        },
        "layers": [0],
        "multiple_choice": False
    },

    "apply": {
        "alg_name": "vector_prompt",
        "layers": [0],
        "multipliers": [1.0],
        "steer_vector_load_dir": "",
        "generation_data": "",
        "generation_output_dir": "logs/",
        "num_responses": 1,
        "generation_data_size": 1
    }
}

# STA
sta_config = {
    "train": {
        "alg_name": "sta",
        "layers": [24],
        "sae_paths": ['/mnt/8t/xhm/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91'],
        "trims": [0.65],
        "mode": "act_and_freq",
        "multiple_choice": False
    },
    "apply": {
        "alg_name": "sta",
        "layers": [24],
        "multipliers": [1.0],
        "steer_vector_load_dir": "",
        "generation_output_dir": "logs/",
        "num_responses": 1,
        "generation_data_size": 1,
        "trims": [0.65],
        "mode": "act_and_freq"
    }
}

# LM Steer
lm_steer_config = {
    "train": {
        # "alg_name": "lm_steer",
        # "adaptor_class": "multiply",
        # "adapted_component": "final_layer",
        # "epsilon": 0.001,
        # "init_var": 0.01,
        # "rank": 1000,
        # "num_steers": 2,
        # "regularization": 0,
        # "optimizer": "Adam",
        # "lr": 0.01,
        # "gamma_mean": 0.99,
        # "n_steps": 100,
        # "seed": 0,
        # "max_length": 256,
        # "batch_size": 16,
        # "log_step": 500,
        # "subset": None,
        # "dummy_steer": 1,
        # "training_steer": 0,
        # "low_resource_mode": None
    },
    "apply": {
        "alg_name": "lm_steer",
        "adaptor_class": "multiply",
        "adapted_component": "final_layer",
        "epsilon": 1e-3,
        "init_var": 1e-2,
        "rank": 1500,
        "num_steers": 2,
        "steer_values": [3,1],
        "steer_vector_load_dir": "/mnt/20t/xuhaoming/EasySteer-simplify/demo/EasySteer/lmsteer_vec"
    }
}

# sae
sae_config = {
    "train": {
        "alg_name": "sae_feature",
        "layer": [20],
        "sae_paths": "/mnt/8t/xhm/models/gemma-scope-9b-it-res/layer_{layer}/width_16k/average_l0_91",
        "release": "gemma-scope-9b-it-res-canonical",
        "sae_id": "layer_{layer}/width_16k/canonical"
    },
    "apply": {
        "alg_name": "sae_feature",
        "layers": [20],
        "multipliers": [1.0],
        "generation_data_size": 1,
        "generation_output_dir": "logs/",
        "num_responses": 1,
        "steer_vector_load_dir": ""
    }
}


demo_config = {
    **common_config,
    "sta_train": sta_config["train"],
    "sta_apply": sta_config["apply"],
    "lm_steer_train": lm_steer_config["train"],
    "lm_steer_apply": lm_steer_config["apply"],
    "caa_train": caa_config["train"],
    "caa_apply": caa_config["apply"],
    "prompt_apply": prompt_config["apply"],
    "autoprompt_apply": autoprompt_config["apply"],
    "vector_prompt_train": vector_prompt_config["train"],
    "vector_prompt_apply": vector_prompt_config["apply"],
    "vector_autoprompt_train": vector_autoprompt_config["train"],
    "vector_autoprompt_apply": vector_autoprompt_config["apply"],
    "sae_feature_train": sae_config["train"],
    "sae_feature_apply": sae_config["apply"]
}

def get_train_hparams(steer_alg, steer_layer):
    train_config = {**common_config, **demo_config[f'{steer_alg}_train']}
    if steer_alg in ['caa', 'vector_prompt', 'vector_autoprompt']:
        train_config['layers'] = [steer_layer]
    if steer_alg in ['sta']:
        train_config['layers'] = [steer_layer]
    # print(train_config)
    # using train_config["alg_name"] rather than steer_alg as for vector_prompt and vector_autoprompt, the alg_name is same
    selected_hparams_class = HYPERPARAMS_CLASS_DICT[train_config["alg_name"]]['train']
    intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(train_config.keys())
    # remove extra fields
    hparams = selected_hparams_class(**{k: train_config[k] for k in intersect_keys})
    
    return hparams

def get_apply_hparams(steer_alg, steer_layer=0, steer_strength=1, prompt=None):
    apply_config = {**common_config, **demo_config[f'{steer_alg}_apply']}
    if steer_alg in ['caa', 'sta', 'vector_prompt', 'vector_autoprompt']:
        apply_config['layers'] = [steer_layer]
        apply_config['multipliers'] = [steer_strength]
    elif steer_alg == 'lm_steer':
        apply_config['steer_values'] = [steer_strength, 1]
    elif steer_alg in ["prompt", "autoprompt"]:
        apply_config['prompt'] = prompt
    elif steer_alg == "sae_feature":
        apply_config['layers'] = [steer_layer]
        apply_config['multipliers'] = [steer_strength]

    selected_hparams_class = HYPERPARAMS_CLASS_DICT[apply_config["alg_name"]]['apply']
    intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(apply_config.keys())
    # remove extra fields
    hparams = selected_hparams_class(**{k: apply_config[k] for k in intersect_keys})
    return hparams