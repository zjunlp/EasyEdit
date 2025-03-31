import os
from omegaconf import OmegaConf
from .alg_dict import HYPERPARAMS_CLASS_DICT, METHODS_CLASS_DICT


def load_generate_vector_hparams(top_cfg):
    hparams_dict = {}
    for i, hparam_path in enumerate(top_cfg.steer_train_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist !"
        method_hparams = OmegaConf.load(hparam_path)
        alg_name = method_hparams['alg_name']
        combined_hparams = {**method_hparams, **top_cfg}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]['train']
        intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        # remove extra fields
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})
        hparams.steer_vector_output_dir = top_cfg.steer_vector_output_dir[i]
        hparams.steer_train_dataset = top_cfg.steer_train_dataset
        hparams_dict[alg_name] = hparams
    return hparams_dict


def load_apply_vector_hparams(top_cfg):
    hparams_dict = {}
    
    # if apply_steer_hparam_paths is not set, return empty dict
    if not hasattr(top_cfg, 'apply_steer_hparam_paths') or not top_cfg.apply_steer_hparam_paths:
        return hparams_dict
        
    for i, hparam_path in enumerate(top_cfg.apply_steer_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist !"
        assert os.path.exists(top_cfg.steer_vector_load_dir[i]), f"Steer vector load path {top_cfg.steer_vector_load_dir[i]} does not exist !"
        method_hparams = OmegaConf.load(hparam_path)
        alg_name = method_hparams['alg_name']
        combined_hparams = {**method_hparams, **top_cfg}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]['apply']
        intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        # remove extra fields
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})
        hparams.steer_vector_load_dir = top_cfg.steer_vector_load_dir[i]
        hparams_dict[alg_name] = hparams

    return hparams_dict

# def apply_steering(hparams_dict, model=None):
#     for alg_name in hparams_dict.keys():
#         if alg_name in METHODS_CLASS_DICT:
#             print(f"Applying {alg_name} vectors to model ...")
#             model = METHODS_CLASS_DICT[alg_name]['apply'](hparams_dict[alg_name], model)
#         else:
#             return NotImplementedError(f"Method {alg_name} not implemented !")
#     return model

# def generate_steering_vector(hparams_dict, dataset):

#     from ..datasets import DatasetLoader
#     loader= DatasetLoader()

#     for alg_name, hparams in hparams_dict.items():
#         if alg_name in METHODS_CLASS_DICT:
#             print(f"Generating {alg_name} vectors ...")
#             if dataset is None:
#                 dataset = loader.load_file(hparams.steer_train_dataset)
#             METHODS_CLASS_DICT[alg_name]['train'](hparams, dataset)
#         else:
#             return NotImplementedError(f"Method {alg_name} not implemented !")