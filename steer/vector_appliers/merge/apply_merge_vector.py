import os
import torch
from ...vector_generators.lm_steer import Hack_no_grad
from .apply_merge_vector_hparam import ApplyMergeVectorHyperParams
         
def reset_merge_vector_layers(model, layers):
    """Reset only the MergeVector activations for specified layers"""
    model=model.model
    for layer in layers:
        if hasattr(model, 'model') and (hasattr(model.model, 'layers') or (hasattr(model.model, 'module') and hasattr(model.model.module, 'layers'))):
            if isinstance(model.model, Hack_no_grad):
                model.model.module.layers[layer].reset(method_name="merge_vector")
            else:
                model.model.layers[layer].reset(method_name="merge_vector")
        elif hasattr(model,'transformer') and hasattr(model.transformer, 'h') or (hasattr(model.transformer, 'module') and hasattr(model.transformer.module, 'h')):  # for GPT models
            if isinstance(model.transformer, Hack_no_grad):
                model.transformer.module.h[layer].reset(method_name="merge_vector")
            else:
                model.transformer.h[layer].reset(method_name="merge_vector")
        else:
            raise NotImplementedError("Failed to reset MergeVector activations")

def apply_merge_vector(hparams: ApplyMergeVectorHyperParams,pipline=None, vector=None):
    from ...models.get_model import get_model
    device = hparams.device
    if pipline is None:
        model, _ = get_model(hparams)
    else:
        model = pipline
    print('Apply MergeVector to model: {}'.format(hparams.model_name_or_path))
    # Reset only MergeVector activations for specified layers
    reset_merge_vector_layers(model, hparams.layers)
    
    layers = hparams.layers
    multipliers = hparams.multipliers
    for layer, multiplier in zip(layers, multipliers):
        print(f"Layer {layer}")

        if vector is not None:
            steering_vector = vector[f'layer_{layer}'].to(device)
            print(f"Steering vector: User input vector for layer_{layer}")
        else:
            vector_path = os.path.join(
                hparams.steer_vector_load_dir, f"layer_{layer}.pt"
            )
            steering_vector = torch.load(vector_path, map_location=device)
            print("Steering vector path: ",vector_path)
        print("Steering vector: ",steering_vector)
        print(f"Multiplier {multiplier}")

        model.set_add_activations(
            layer, multiplier * steering_vector, method_name="merge_vector"
        )
    return model

# def eval_merge_vector(hparams: ApplyMergeVectorHyperParams):
#     dataset = GenerationDataset()
#     if "toxigen" in hparams.eval_data_name:
#         get_data=dataset.get_data_for_toxigen
#     elif "realtoxicity" in hparams.eval_data_name:
#         get_data=dataset.get_data_for_realtoxicity
#     elif "gsm" in hparams.eval_data_name:
#         get_data=dataset.get_data_for_gsm
#     else:
#         get_data=dataset.get_data_for_merge_vector
    
#     test_dataset = get_data(
#         data_path=hparams.data_path,
#         data_name=hparams.eval_data_name,
#         split="test",
#     )
    
#     questions = test_dataset["question"]
#     if "toxigen" in hparams.eval_data_name:
#         label = test_dataset["label"]
#     elif "gsm" in hparams.eval_data_name:
#         answers = test_dataset["answer"]
#     elif "exaggerated-safety" in hparams.eval_data_name:
#         ids = test_dataset["id"]
#         tys = test_dataset["type"]

#     model = apply_merge_vector(hparams)
#     device = model.device
#     tokenizer = model.tokenizer
#     prompt_tokens_list = []

#     for i in range(len(test_dataset)):
#         ques = test_dataset[i]["question"]

#         if not ques: continue
#         ques_tokens = tokenizer.encode(ques, return_tensors="pt")    
#         # use question as the final prompt, for testing the inference process
#         prompt_tokens_list.append(ques_tokens.to(device))

#     directory = os.path.dirname(hparams.output_file)  
#     if not os.path.exists(directory):  
#         os.makedirs(directory)
    
#     preds = []
#     print("max_new_tokens: ", hparams.max_new_tokens)
#     for prompt_tokens in tqdm(prompt_tokens_list, desc=f"Generating... layer-{hparams.layers} multiplied by {hparams.multipliers}"):
#         prompt_tokens = prompt_tokens.to(device)  
#         output = model.model.generate(prompt_tokens, max_new_tokens=hparams.max_new_tokens)
#         output = output[:,prompt_tokens.shape[-1]:]
#         output = tokenizer.batch_decode(output)
#         preds.append(output[0])
#     print("Without clean_preds!!!")

#     if "toxigen" in hparams.eval_data_name:
#         results = [
#             {"question": questions[idx], "pred": preds[idx], "label": label[idx]} for idx in range(len(preds))
#         ]
#     elif "gsm" in hparams.eval_data_name:
#         results = [
#             {"question": questions[idx], "answer": answers[idx], "pred": preds[idx]} for idx in range(len(preds))
#         ]
#     elif "exaggerated-safety" in hparams.eval_data_name:
#         results = [
#             {"id":ids[idx], "type":tys[idx], "question": questions[idx], "pred": preds[idx]} for idx in range(len(preds))
#         ]
#     else:
#         results = [
#             {"question": questions[idx], "pred": preds[idx]} for idx in range(len(preds))
#         ]

#     json.dump(results, open(hparams.output_file, 'w'), indent=4, ensure_ascii=False)
        