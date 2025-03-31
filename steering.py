from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.datasets import prepare_train_dataset, prepare_generation_datasets
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.2',config_path='./hparams/Steer', config_name='config.yaml')
def main(top_cfg: DictConfig):
    print("Global Config:", top_cfg, "\n")
    
    # Vector Generation
    # You can customize your own inputs
    train_datasets = {
        'your_dataset_name':[
            {'question': 'What do you like to do in your free time?', 
            'matching':'我喜欢阅读、听音乐和旅行。', 
            'not_matching':'I like reading, listening to music, and traveling.'}, 
        ]
    }
    # Or use the datasets from config.yaml
    # train_datasets = prepare_train_datasets(top_cfg)
    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.generate_vectors(train_datasets)

    # Vector Application
    vector_applier = BaseVectorApplier(top_cfg)
    print(vectors)
    for dataset in vectors.keys():
        print(f"Applying  {dataset} vectors to model ...")
        vector_applier.apply_vectors(vectors[dataset])

    # vector_applier.apply_vectors()

    # Result Generation
    # You can customize your own inputs
    # generation_datasets={'your_dataset_name':[{'input':'How do you feel about the recent changes at work?'},{'input':'how are you'}]}
    
    # Or use the datasets from config.yaml
    generation_datasets = prepare_generation_datasets(top_cfg)
    
    # # # Method 1: Use parameters from config.yaml
    vector_applier.generate(generation_datasets)
    
    # Method 2: Use parameters from function (uncomment to use)
    # generation_params = get_generation_params()
    # vector_applier.generate(generation_datasets, **generation_params)
    
    # Resets the model to its initial state, clearing any modifications.
    # vector_applier.model.reset_all()

if __name__=='__main__':
    main()
