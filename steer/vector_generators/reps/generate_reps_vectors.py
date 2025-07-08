import os
import argparse
import torch
from tqdm import tqdm
from .generate_reps_vector_hparams import RePSHyperParams
from .utils import get_grouped_data_by_concept_id, get_prefix_length, load_state, prepare_groups, save_state
from steer.trainer.RepsVectorTrainer import RepsVectorTrainer

def generate_reps_vectors(args:RePSHyperParams, dataset, model = None):
    from ...models.get_model import get_model
    # Whether to delete the model after the function
    del_model = True

    # Load grouped data by concept_id, key is concept_id, value is da
    concept_grouped_data = get_grouped_data_by_concept_id(dataset)
    
    
    print(f"[INFO] Total number of concept df loaded: {len(concept_grouped_data)}")
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
    model.tokenizer.model_max_length = 512
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

    prefix_length = 1 # prefix is default to 1 for all models due to theBOS token.
    if args.use_chat_template:
        prefix_length = get_prefix_length(tokenizer)
        print(f"[WARNING] Chat model prefix length: {prefix_length}")

    state = load_state(args.steer_vector_output_dir)
    last_concept_id = state.get("last_concept_id", None) if state else None
    print(f"[WARNING] last concept_id processed: {last_concept_id}")

<<<<<<< HEAD
    vec = None
=======
>>>>>>> 2a46a1c5 (Forward Still Bug)
    for concept_id, concept_groups in concept_grouped_data:
        concept_id = int(concept_id) 

        # if the concept has been processed, skip it
        if last_concept_id is not None and concept_id <= last_concept_id:
            print(f"[WARNING] Skip concept_id {concept_id}, because it has been processed")
            continue
        
        print(f"[WARNING] Start to train RePS for concept_id {concept_id}")

        concept = concept_groups[0]["concept"]
        print(f"[WARNING] Using concept '{concept}' to train RePS")
        
        # Init the trainer
        benchmark_model = RepsVectorTrainer(
            model=model,
            layers=args.layers,
            hparams=args 
        )
        # get the low rank dimension,  set to 1 as a vector
        low_rank_dimension = args.low_rank_dimension if args.low_rank_dimension else 1
        
        # incorporate the intervention to the model
        benchmark_model.make_model(
            mode="train",
            embed_dim=model.model.config.hidden_size,
            low_rank_dimension=low_rank_dimension,
            dtype=model.torch_dtype,
            intervention_type=args.intervention_type, 
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
        )

        benchmark_model.train(prepared_groups, **training_kwargs)

        # get the trained vectors
<<<<<<< HEAD
        vec = benchmark_model.model.steer_vector.proj.weight.data.clone().cpu()
=======
        vectors = {}
        for layer in args.layers:
            vec = benchmark_model.model.steer_vector.proj.weight.data.clone().cpu()
            vectors[f"layer_{layer}"] = vec
>>>>>>> 2a46a1c5 (Forward Still Bug)
        
        # save the current state
        current_state = {'last_concept_id': concept_id}
        save_state(args.steer_vector_output_dir, current_state)
        
        # save the vectors and the model
        if args.save_vectors:
            concept_dir = os.path.join(args.steer_vector_output_dir, f"concept_{concept}")
            os.makedirs(concept_dir, exist_ok=True)
<<<<<<< HEAD
            torch.save(vec, os.path.join(concept_dir, f"concept_{concept}.pt"))
            # benchmark_model.save(concept_dir, model_name=f"{args.model_name_or_path}")
=======
            torch.save(vectors, os.path.join(concept_dir, f"layer_{args.layers}.pt"))
            benchmark_model.save(concept_dir, model_name=f"{args.model_name_or_path}")
>>>>>>> 2a46a1c5 (Forward Still Bug)
        
        # clean the memory
        del benchmark_model
        torch.cuda.empty_cache()

    if del_model:
        model.model.to('cpu')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
<<<<<<< HEAD
    if vec:
        return vec
=======
    
    return vectors
>>>>>>> 2a46a1c5 (Forward Still Bug)
