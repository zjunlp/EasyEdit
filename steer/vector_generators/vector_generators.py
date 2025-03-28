from omegaconf import DictConfig
import os

class BaseVectorGenerator:
    """Base vector generator for all methods"""

    # @classmethod
    # def from_hparams(cls, hparams: HyperParams):
    #     return cls(hparams)

    def __init__(self, top_cfg: DictConfig):
        from ..utils import load_generate_vector_hparams
        self.hparams_dict = load_generate_vector_hparams(top_cfg)
        for alg_name, hparams in self.hparams_dict.items():
            print(f"{alg_name.upper()} Generator Hyperparameters:\n{hparams}")
        

    def generate_vectors(self, datasets = None,):
        
        from ..utils.seed import set_seed
        from ..utils.alg_dict import METHODS_CLASS_DICT
        # generate_steering_vector(self.hparams_dict,dataset)
        assert datasets is not None, "Please provide datasets!"
        generated_vectors = {}
        for dataset_name in datasets:
            generated_vectors[ dataset_name ] = {}
            for alg_name, hparams in self.hparams_dict.items():
                if alg_name in METHODS_CLASS_DICT:
                    # load the dataset from the hparams path if not provided
                    # if dataset is None:
                    #     loader= DatasetLoader()
                    #     dataset= loader.load_file(hparams.steer_train_dataset, split='train') 
                    set_seed(hparams.seed)
                    steer_vector_output_dir = hparams.steer_vector_output_dir
                    hparams.steer_vector_output_dir = os.path.join(hparams.steer_vector_output_dir, dataset_name)
                    
                    print(f"Generating {alg_name} vectors ...")
                    if alg_name in ['lm_steer', 'caa', 'vector_prompt', 'sta']:
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams, datasets[dataset_name])
                    else:
                        vectors = METHODS_CLASS_DICT[alg_name]['train'](hparams)
                    generated_vectors[dataset_name][alg_name] = vectors
                    print(f"Saving vectors to {hparams.steer_vector_output_dir} ...\n")
                    hparams.steer_vector_output_dir = steer_vector_output_dir
                    
                else:
                    raise NotImplementedError(f"Method {alg_name} not implemented !")
                
        return generated_vectors
                
