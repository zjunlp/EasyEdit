import hydra
from omegaconf import OmegaConf, DictConfig
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.datasets import prepare_train_datasets

@hydra.main(version_base='1.2',config_path='./hparams/Steer', config_name='vector_generate.yaml')
def main(top_cfg: DictConfig):
    # top_cfg = OmegaConf.load(cfg_path)
    print("Global Config:", top_cfg, "\n")
    
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

    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.generate_vectors(datasets)
    print("Generated Vectors:", vectors)

if __name__ == '__main__':
    main()
    

    # hparams_dict = load_generate_vector_hparams(top_cfg)
    # print("Hparams Dict:", hparams_dict)
    # generate_steering_vector(hparams_dict)
    # generate_lm_steer_vector(hparams_dict)
    # generate_caa_vector(hparams_dict)
    # generate_vector_prompt_vector(hparams_dict)


# def generate_lm_steer_vector(hparams_dict):
#     assert 'lm_steer' in hparams_dict, "Please provide lmsteer hparams path !"
#     hparams = hparams_dict['lm_steer']
#     generate_lm_steer_delta(hparams)
    
# def generate_caa_vector(hparams_dict):
#     assert 'caa' in hparams_dict, "Please provide caa hparams path !"
#     hparams = hparams_dict['caa']
#     generate_caa_vectors(hparams)
    
# def generate_vector_prompt_vector(hparams_dict):
#     assert 'vector_prompt' in hparams_dict, "Please provide vector_prompt hparams path !"
#     hparams = hparams_dict['vector_prompt']
#     generate_vector_prompt_vectors(hparams)
    
# def generate_sae_feature_vector(hparams_dict):
#     assert 'sae_feature' in hparams_dict, "Please provide sae_feature hparams path !"
#     hparams = hparams_dict['sae_feature']
#     generate_sae_feature_vectors(hparams)

# def generate_sta_vector(hparams_dict):
#     assert 'sta' in hparams_dict, "Please provide sta hparams path !"
#     hparams = hparams_dict['sta']
#     generate_sta_vectors(hparams)

