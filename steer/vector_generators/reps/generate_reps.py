import os
import argparse
import torch
import json
from tqdm import tqdm
from .generate_reps_hparams import RePSHyperParams
from .utils import get_prefix_length, load_state, prepare_groups, save_state
from steer.trainer.RepsVectorTrainer import RepsVectorTrainer


def generate_reps(args: RePSHyperParams, dataset, model = None, dataset_name = None):
    """
    Generate RePS vectors from dataset.
    
    Args:
        args: RePS hyperparameters
        dataset: Either List[Dict] (single concept) or List[Tuple] (multi concept grouped data)
        model: Pre-loaded model (optional)
    
    Returns:
        Generated vector tensor (single concept) or vector dict (multi concept)
    """
    from ...models.get_model import get_model
    
    # Determine dataset type by checking data structure
    if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], tuple):
        # Multi-concept dataset: List[Tuple[concept_id, concept_data]]
        concept_grouped_data = dataset
        print(f"[INFO] Processing multi-concept dataset with {len(concept_grouped_data)} concepts")
        is_multi_concept = True
    elif isinstance(dataset, list):
        # Single-concept dataset: List[Dict] - convert to grouped format for uniform processing
        dataset_name = "single_concept" if dataset_name is None else dataset_name
        concept_grouped_data = [(dataset_name, dataset)]
        print(f"[INFO] Processing {dataset_name} dataset with {len(dataset)} items")
        is_multi_concept = False
    else:
        raise ValueError(f"Unsupported dataset format: {type(dataset)}")
    
    # Whether to delete the model after the function
    del_model = True

    
    print(f"[INFO] Total number of concepts loaded: {len(concept_grouped_data)}")
    if args.max_concepts:
        print(f"[WARNING] Only processing {args.max_concepts} concepts")
        concept_grouped_data = concept_grouped_data[:args.max_concepts]

    if model is None:
        model, tokenizer = get_model(args)
    else:
        del_model = False
        model, tokenizer = model, model.tokenizer
        model.hparams = args

    
    # Load model instance onto device
    if model.torch_dtype == torch.bfloat16:
        print(f"[WARNING] Using bfloat16 for model {args.model_name_or_path}")
        
    model.tokenizer.padding_side = "right"
    if args.output_length and args.output_length > 0:
        model.tokenizer.model_max_length = args.output_length

    model.model = model.model.eval()
    model.model.to(args.device)
    # is_chat_model = True if args.model_name in CHAT_MODELS else False
    
   
    if model.tokenizer.unk_token == None and model.tokenizer.pad_token == None:
        # raw llama3
        print("[INFO] adding a special padding token...")
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        need_resize = True
    else:
        need_resize = False
    if need_resize:
       model.model.resize_token_embeddings(len(model.tokenizer))

    all_vectors = {}
    for layer in args.layers:
        prefix_length = 1 # prefix is default to 1 for all models due to theBOS token.
        if args.use_chat_template:
            prefix_length = get_prefix_length(tokenizer)
            print(f"[WARNING] Chat model prefix length: {prefix_length}")
        
        output_dir = os.path.join(
            args.steer_vector_output_dir,
            f"reps_{args.intervention_method}"
        )
        if args.save_vectors:
            os.makedirs(output_dir, exist_ok=True)

        state = load_state(output_dir, state_file=f"train_state_layer_{layer}.pkl")
        last_concept_id = state.get("last_concept_id", None) if state else None
        print(f"[WARNING] last concept_id processed: {last_concept_id}")

        # For multi-concept datasets, store all vectors and their concept info
        datas_dict = {} if is_multi_concept else None
        concept_info_dict = {} if is_multi_concept else None  # Store concept descriptions
        data_state = None
        
        for concept_key, concept_groups in concept_grouped_data:
            # Handle both numeric concept_id and string dataset names
            if is_multi_concept:
                # For multi-concept datasets, try to convert concept_key to int
                concept_id = int(concept_key)
                concept_key_str = str(concept_key)
            else:
                concept_id = 0  # Default for single-concept datasets
                concept_key_str = str(concept_key)

            # if the concept has been processed, skip it (only for multi-concept datasets)
            if args.save_vectors and is_multi_concept and last_concept_id is not None and concept_id <= last_concept_id:
                print(f"[WARNING] Skip concept_id {concept_id}, because it has been processed")
                continue
            
            print(f"[INFO] Start to train RePS for concept: {concept_key_str}")

            # Get concept name
            if is_multi_concept:
                concept = concept_groups[0].get("concept", concept_groups[0].get("output_concept", str(concept_key)))
                concept_info_dict[concept_id] = concept
            else:
                concept = concept_key_str
            print(f"[INFO] Using concept '{concept}' to train RePS")
            
            # Store concept info for multi-concept datasets
            
            
            # Init the trainer
            benchmark_model = RepsVectorTrainer(
                model=model,
                layers=[layer],
                hparams=args
            )
            # get the low rank dimension,  set to 1 as a vector
            low_rank_dimension = args.low_rank_dimension if args.low_rank_dimension else 1
            
            # incorporate the intervention to the model
            benchmark_model.make_model(
                mode="train",
                input_dim=model.model.config.hidden_size if args.intervention_components != "mlp_mid" else model.model.config.intermediate_size,
                embed_dim=model.model.config.hidden_size,
                intervention_components=args.intervention_components,
                low_rank_dimension=low_rank_dimension,
                dtype=model.torch_dtype,
                intervention_type=args.intervention_type, 
                intervention_method=args.intervention_method,
                concept_id=concept_id,
                dump_dir=args.steer_vector_output_dir,
                model_params=args,
                dropout=args.dropout,
                intervention_positions_dropout=args.intervention_positions_dropout,
                preference_pairs=args.preference_pairs,
            )

            benchmark_model.model.steer_vector.to(model.torch_dtype)
            
            # prepare the training parameters
            training_kwargs = {
                "prefix_length": prefix_length,
                "positions": args.intervention_positions, 
                "exclude_bos": args.exclude_bos,

                "preference_pairs": args.preference_pairs,
                "steering_prompt_type": args.steering_prompt_type,
                "substraction_type": args.substraction_type,
                "use_chat_template": args.use_chat_template,
            }
            # prepare the data
            prepared_groups = prepare_groups(
                concept_groups, 
                concept, 
                model.tokenizer, 
                use_chat_template=args.use_chat_template,
                model_name_or_path=args.model_name_or_path,
                max_num_of_examples=args.max_num_of_examples,
                steering_prompt_type=args.steering_prompt_type,
                is_select_category=is_multi_concept
            )
            benchmark_model.train(prepared_groups, **training_kwargs)

            if args.intervention_method == "vector":
                data_state = benchmark_model.model.steer_vector.proj.weight.data.clone().cpu()
                data_state = data_state.squeeze(0)
            elif args.intervention_method == "lora":
                lora_module = benchmark_model.model.steer_vector
                data_state = {
                    "lora_A": lora_module.lora_A.detach().cpu(),
                    "lora_B": lora_module.lora_B.detach().cpu(),
                    "r": lora_module.r,
                    "alpha": lora_module.lora_alpha,
                    "intervention_components": lora_module.intervention_components,
                    "concept_id": concept_id,
                }
            elif args.intervention_method == "local_weight":
                local_weight_module = benchmark_model.model.steer_vector
                data_state = {
                    "weight": local_weight_module.delta_weight.detach().cpu(),
                    "bias": local_weight_module.delta_bias.detach().cpu(),
                    "intervention_components": local_weight_module.intervention_components,
                    "concept_id": concept_id,
                }
            # Store vector for multi-concept datasets
            if is_multi_concept:
                datas_dict[concept_id] = data_state
            
            # save the vectors immediately after training each concept
            if args.save_vectors:
                if is_multi_concept:
                    current_state = {'last_concept_id': concept_id}
                    save_state(output_dir, current_state, state_file=f"train_state_layer_{layer}.pkl")
                    
                    # For multi-concept datasets, append to the same pt file
                    vector_filename = f"layer_{layer}.pt"
                    vector_filepath = os.path.join(output_dir, vector_filename)
                    
                    # Load existing vectors if file exists, otherwise create new list
                    if os.path.exists(vector_filepath):
                        existing_vectors = torch.load(vector_filepath)
                        # Convert to list if it's a tensor
                        if isinstance(existing_vectors, torch.Tensor):
                            existing_vectors = [existing_vectors[i] for i in range(existing_vectors.shape[0])]
                        elif not isinstance(existing_vectors, list):
                            existing_vectors = [existing_vectors]
                    else:
                        if args.intervention_method == "vector":
                            existing_vectors = []
                        else: 
                            existing_vectors = {}
                    
                    # Append new vector and Save updated vectors as stacked tensor
                    if args.intervention_method == "vector":
                        existing_vectors.append(data_state)
                        combined_vectors = torch.stack(existing_vectors, dim=0)
                    else:
                        existing_vectors[concept_id] = data_state
                        combined_vectors = existing_vectors

                    torch.save(combined_vectors, vector_filepath)
                    
                    # Also save/append metadata for this concept
                    metadata = {
                        "concept_id": concept_id,
                        "components": args.intervention_components if args.components else None,
                        "rank": args.low_rank_dimension,
                        "alpha": args.lora_alpha if args.alpha else None,
                        "concept": concept,  # Use the concept name we got earlier
                        "ref": f"vector_index_{len(existing_vectors)-1}"  # Index in the stacked tensor
                    }
                    
                    metadata_file = os.path.join(output_dir, f"metadata_layer_{layer}.jsonl")
                    with open(metadata_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata) + '\n')
                    
                    print(f"[INFO] Saved vector for concept {concept_id} to {vector_filename} (total: {len(existing_vectors)} vectors)")
                else:
                    # For single-concept datasets, use original filename
                    torch.save(data_state, os.path.join(output_dir, f"layer_{layer}.pt"))
                    print(f"[INFO] Saved single-concept vector to layer_{layer}.pt")

            
            # clean the memory
            del benchmark_model
            torch.cuda.empty_cache()

        if is_multi_concept:
            # After processing all concepts, return the vectors dict
            all_vectors[f"layer_{layer}"] = datas_dict
        else:
            all_vectors[f"layer_{layer}"] = data_state

    if del_model:
        model.model.to('cpu')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return appropriate result based on dataset type
    print(f"[INFO] Generated RePS {args.intervention_method} : {all_vectors}")
    if is_multi_concept:
        return all_vectors
    else:
        # For single-concept datasets, return the single vector
        return all_vectors
