from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.datasets import prepare_train_dataset
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.2',config_path='./hparams/Steer', config_name='config_test_reps.yaml')
def main(top_cfg: DictConfig):

    print("Global Config:", top_cfg, "\n")
    # Prepare datasets
    train_datasets = prepare_train_dataset(top_cfg)

    # generation_datasets = prepare_generation_datasets(top_cfg)

    # Generate Steering Vectors
    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.generate_vectors(train_datasets)

    # Apply Vectors to Model 
    vector_applier = BaseVectorApplier(top_cfg)
    print(vectors)
    for dataset in vectors.keys():
        print(f"Applying  {dataset} vectors to model ...")
        vector_applier.apply_vectors(vectors[dataset])

    # vector_applier.apply_vectors()

    # Result Generation
    # vector_applier.generate(generation_datasets)

    # # Resets the model to its initial state, clearing any modifications.
    # vector_applier.model.reset_all()

if __name__=='__main__':
    main()
