import os
import torch
from .generate_merge_vector_hparams import MergeVectorHyperParams
from .merge_helpers import merge_steer_vectors, merge_linear



method_dict = {
    'linear': None,
    'ties': {
        'consensus_method': 'sum',
        'sparsification_method': 'magnitude',
        'normalize': True,
        'rescale_norm': None
    },
    'dare_ties': {
        'consensus_method': 'sum',
        'sparsification_method': 'random',
        'normalize': False,
        'rescale_norm': 'l1'
    },
    'dare_linear': {
        'consensus_method': None,
        'sparsification_method': 'random',
        'normalize': False,
        'rescale_norm': 'l1'
    },
}

def generate_merge_vector(hparams: MergeVectorHyperParams):
    """
    Generate merged vector based on specified hyperparameters.
    """
    from ...utils.alg_dict import DTYPES_DICT
    # load the vectors
    vectors = []
    for vector_path in hparams.vector_paths:
        assert os.path.exists(vector_path), f"Vector path {vector_path} does not exist"
        vector = torch.load(vector_path, map_location=hparams.device)
        vectors.append(vector)
        
    weights = hparams.weights
    densities = hparams.densities
    mask_dtype = DTYPES_DICT.get(hparams.torch_dtype, torch.float32)
    kwargs = method_dict.get(hparams.method, {})
    
    # merge the vectors
    if hparams.method in ["ties", "dare_ties", "dare_linear"]:
        merged_vector = merge_steer_vectors(
            vectors=vectors,
            weights=weights,
            densities=densities,
            mask_dtype=mask_dtype,
            **kwargs
        )
    elif hparams.method == "linear":
        merged_vector = merge_linear(
            vectors=vectors,
            weights=weights,
            normalize=hparams.normalize
        )
    else:
        raise ValueError(f"Invalid method: {hparams.method}")

    # save the merged vector
    base_name = os.path.basename(hparams.vector_paths[0]).split('.')[0]
    save_path = os.path.join(hparams.steer_vector_output_dir, 'merge_vector', f"{base_name}_merged_vector.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hparams.save_vectors is True:
        torch.save(merged_vector, save_path)
    return {f"{base_name}_merged_vector": merged_vector}

if __name__ == "__main__":
    # Test
    vectors = [
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1.0, 2.0, 3.0],dtype=torch.float32)
    ]
    weights = [1.0,1.0]  
    densities = [1.0,1.0]  
    kwargs = method_dict.get('ties')
    # Merge vectors
    merged_vector = merge_steer_vectors(
        vectors=vectors,
        weights=weights,
        densities=densities,
        mask_dtype=torch.float32,
        **kwargs
    )
    print("Merged steering vector:", merged_vector)