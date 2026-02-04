from omegaconf import DictConfig
import os

REQUIRED_DATASET_METHODS=['lm_steer', 'caa', 'vector_prompt', 'sta', 'reps', 'sft', 'spilt']

class BaseVectorGenerator:
    """Base vector generator for all methods"""

    # @classmethod
    # def from_hparams(cls, hparams: HyperParams):
    #     return cls(hparams)

    def __init__(self, top_cfg: DictConfig):
        from ..utils import load_generate_vector_hparams
        self.hparams_dict = load_generate_vector_hparams(top_cfg)
        for key, hparams in self.hparams_dict.items():
            print(f"{key.upper()} Generator Hyperparameters:\n{hparams}")
        

    def generate_vectors(self, datasets = None,):
        
        from ..utils.seed import set_seed
        from ..utils.alg_dict import METHODS_CLASS_DICT,VLLM_SUPPORTED_METHODS
        # generate_steering_vector(self.hparams_dict,dataset)
        assert datasets is not None, "Please provide datasets!"
        generated_vectors = {}
        for dataset_name in datasets:
            generated_vectors[ dataset_name ] = {}
            for key, hparams in self.hparams_dict.items():
                alg_name = hparams.alg_name
                if alg_name in METHODS_CLASS_DICT:
                    set_seed(hparams.seed)
                    #build vector save path
                    steer_vector_output_dir = hparams.steer_vector_output_dir
                    hparams.steer_vector_output_dir = os.path.join(hparams.steer_vector_output_dir, dataset_name)
                    
                    now_path = os.path.join(hparams.steer_vector_output_dir, alg_name + '_vector')
                    if os.path.exists(now_path) and hparams.save_vectors:
                        print('\033[1;34mVectors save path already exists! The vector will be overwritten!\033[0m')
                    
                    print(f"Generating {key} vectors ...")
                    if alg_name in REQUIRED_DATASET_METHODS:
                        if alg_name not in VLLM_SUPPORTED_METHODS:
                            hparams.vllm_enable = False
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams, datasets[dataset_name], dataset_name=dataset_name)
                    else:
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams)
                    generated_vectors[dataset_name][key] = vectors
                    if hparams.save_vectors:
                        print(f"Saving vectors to {now_path} ...")
                    else:
                        print(f"Not saving {key} vectors ...")
                    hparams.steer_vector_output_dir = steer_vector_output_dir
                    
                else:
                    raise NotImplementedError(f"Method {alg_name} not implemented !")
                
        return generated_vectors

    def multimodal_generate_vectors(self, datasets = None,):
        from ..utils.seed import set_seed
        from ..utils.alg_dict import METHODS_CLASS_DICT
        assert datasets is not None, "Please provide datasets!"
        generated_vectors = {}
        for dataset_name in datasets:
            generated_vectors[dataset_name] = {}
            for key, hparams in self.hparams_dict.items():
                alg_name = hparams.alg_name
                if alg_name in METHODS_CLASS_DICT:
                    set_seed(hparams.seed)
                    steer_vector_output_dir = hparams.steer_vector_output_dir
                    hparams.steer_vector_output_dir = os.path.join(hparams.steer_vector_output_dir, dataset_name)
                    now_path = os.path.join(hparams.steer_vector_output_dir, alg_name + '_vector')
                    if os.path.exists(now_path) and hparams.save_vectors:
                        print('\u001b[1;34mVectors save path already exists! The vector will be overwritten!\u001b[0m')
                    print(f"Generating {key} vectors ...")
                    # For multimodal, pass dataset when required (same as text)
                    if alg_name in REQUIRED_DATASET_METHODS:
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams, datasets[dataset_name], dataset_name=dataset_name)
                    else:
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams)
                    generated_vectors[dataset_name][key] = vectors
                    if hparams.save_vectors:
                        print(f"Saving vectors to {now_path} ...")
                    else:
                        print(f"Not saving {key} vectors ...")
                    hparams.steer_vector_output_dir = steer_vector_output_dir
                else:
                    raise NotImplementedError(f"Method {alg_name} not implemented !")
        return generated_vectors
