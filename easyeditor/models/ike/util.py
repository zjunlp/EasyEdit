from sentence_transformers import SentenceTransformer
import pickle
from torch.utils.data import Dataset
import os
from .ike_hparams import IKEHyperParams


def encode_ike_facts(sentence_model: SentenceTransformer, ds: Dataset, hparams: IKEHyperParams):

    sentences = []
    for i, train_data in enumerate(ds):
        new_fact = train_data['prompt'] + ' ' + train_data['target_new']
        target_new = train_data['target_new']
        paraphrases = train_data['rephrase_prompt']
        neighbors = train_data['locality_prompt']
        neighbors_ans = train_data['locality_ground_truth']
        sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")
        sentences.append(f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n")
        sentences.append(f"New Fact: {new_fact}\nPrompt: {neighbors} {neighbors_ans}\n\n")

    embeddings = sentence_model.encode(sentences)
    base_path = f'{hparams.results_dir}/{hparams.alg_name}/embedding'
    os.makedirs(base_path, exist_ok=True)
    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{base_path}/{safe_model_name}_{type(ds).__name__}_{len(ds)}.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)
