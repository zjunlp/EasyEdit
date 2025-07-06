import pickle
import os
from pathlib import Path
<<<<<<< HEAD
from steer.utils.templates import model_supports_system_prompt
STATE_FILE = "train_state.pkl"

def get_grouped_data_by_concept_id(dataset):
    # Group data by concept_id
    concept_groups = {}
=======

STATE_FILE = "train_state.pkl"



def get_grouped_data_by_concept_id(dataset):
    # Group data by concept_id
    concept_groups = {}
    
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    # Group records by concept_id
    for record in dataset:
        concept_id = record['concept_id']
        if concept_id >= 0:
            if concept_id not in concept_groups:
                concept_groups[concept_id] = []
            concept_groups[concept_id].append(record)
    
    # Get sorted concept_ids and create list of tuples
    concept_ids = sorted(concept_groups.keys())
    grouped_data = []
    
    for concept_id in concept_ids:
        # print(f"Processing concept_id {concept_id}")
        grouped_data.append((concept_id, concept_groups[concept_id]))
    
    return grouped_data

def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
        print("Detecting sequence a:", tokens_a)
        print("Detecting sequence b:", tokens_b)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True)
        prefix_length = len(tokens)
    return prefix_length

<<<<<<< HEAD
def get_suffix_length(tokenizer):
    message_a = [{"role": "user", "content": "1"}]
    message_b = [{"role": "user", "content": "2"}]
    tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
    tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])

def prepare_groups(
    prepared_groups, concept, tokenizer, 
    use_chat_template, model_name_or_path, 
    max_num_of_examples=None, steering_prompt_type="prepend",
    ):
    
    suffix_length, suffix_str = get_suffix_length(tokenizer)
    print(f"Suffix length for {model_name_or_path}: {suffix_length}, Suffix string: {suffix_str}")

    positive_data = [item for item in prepared_groups 
                    if item.get("output_concept") == concept and item.get("category") == "positive"]

    # limit the number of examples
    if max_num_of_examples:
        positive_data = positive_data[:max_num_of_examples // 2]
    
    all_data = positive_data
        
    if use_chat_template:
        system_messages = []
        if model_supports_system_prompt(model_name_or_path):
            system_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        def apply_input_template(item, column_name):
            messages = system_messages + [{"role": "user", "content": item[column_name]}]
            nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
            return tokenizer.decode(nobos)
        
        def apply_output_template(text):
            return text + suffix_str
        
        processed_data = []
        for item in all_data:
            processed_item = item.copy()
            processed_item["question"] = apply_input_template(item, "question")
            output_columns = ["output", "matching", "not_matching", 
                            "prepend_steered_output", "blend_in_steered_output"]
            for column in output_columns:
                if column in processed_item:
                    processed_item[column] = apply_output_template(processed_item[column])
            processed_data.append(processed_item)
            
        if processed_data:
            print("\n=== Sample Row Data ===")
            sample_item = processed_data[0]
            for key, value in sample_item.items():
                print(f"\n{key}:")
                print("-" * (len(key) + 1))
                print(f"{value}")
            print("=====================\n")
            
        return processed_data
    else:
        # if not use chat template, return the original data
        return all_data
=======
def prepare_df(
    original_df, negative_df, concept, metadata, tokenizer, 
    binarize, train_on_negative, is_chat_model, output_length, model_name, 
    max_num_of_examples=None, use_dpo_loss=False, steering_prompt_type="prepend",
    keep_orig_axbench_format=False):
    
    suffix_length, suffix_str = get_suffix_length(tokenizer)
    print(f"Suffix length for {model_name}: {suffix_length}, Suffix string: {suffix_str}")
    genre = metadata["concept_genres_map"][concept][0]
    # assign input and output containing concept with 1, otherwise 0
    positive_df = original_df[(original_df["output_concept"] == concept) & (original_df["category"] == "positive")]
    negative_df = negative_df[(negative_df["concept_genre"] == genre)]
    if max_num_of_examples:
        positive_df = positive_df.head(max_num_of_examples // 2)
        negative_df = negative_df.head(max_num_of_examples // 2)
    if binarize:
        if is_chat_model:
            if model_name in HAS_SYSTEM_PROMPT_MODELS:
                def apply_chat_template(row):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}
                    ]
                    nobos = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True)[1:-suffix_length]
                    return tokenizer.decode(nobos)
                positive_df = positive_df.copy()
                negative_df = negative_df.copy()
                positive_df['combined'] = positive_df.apply(apply_chat_template, axis=1)
                negative_df['combined'] = negative_df.apply(apply_chat_template, axis=1)
            else:
                def apply_chat_template(row):
                    messages = [
                        {"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}
                    ]
                    nobos = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True)[1:-suffix_length]
                    return tokenizer.decode(nobos)
                positive_df = positive_df.copy()
                negative_df = negative_df.copy()
                positive_df['combined'] = positive_df.apply(apply_chat_template, axis=1)
                negative_df['combined'] = negative_df.apply(apply_chat_template, axis=1)
        else:
            positive_df = positive_df.copy()
            negative_df = negative_df.copy()
            positive_df['combined'] = positive_df['input'] + positive_df['output']
            negative_df['combined'] = negative_df['input'] + negative_df['output']
        positive_df = pd.DataFrame(positive_df[['combined']]).rename(columns={'combined': 'input'})
        negative_df = pd.DataFrame(negative_df[['combined']]).rename(columns={'combined': 'input'})
        positive_df["labels"] = 1
        negative_df["labels"] = 0
        return pd.concat([positive_df, negative_df], axis=0)
    else:
        # if not binarizing, we need to apply the chat template to the input. It becomes a standard instruction tuning task.
        if not use_dpo_loss and train_on_negative:
            all_df = pd.concat([positive_df, negative_df], axis=0)
        else:
            # for DPO, we only use positive examples.
            all_df = positive_df
        if is_chat_model:
            system_messages = []
            if model_name in HAS_SYSTEM_PROMPT_MODELS:
                system_messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            def apply_chat_template(df, column_name):
                def template_function(row):
                    messages = system_messages + [{"role": "user", "content": row[column_name]}]
                    nobos = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[1:]
                    return tokenizer.decode(nobos)
                df[column_name] = df.apply(template_function, axis=1)

            apply_chat_template(all_df, "input")
            if use_dpo_loss:
                if f"{steering_prompt_type}_steered_input" in all_df.columns:
                    apply_chat_template(all_df, f"{steering_prompt_type}_steered_input")

            # Add EOS prefix tokens by default. The truncation at data collator will take care of the rest.
            def apply_output_template(df, column_name):
                def template_function(row):
                    return row[column_name] + suffix_str
                df[column_name] = df.apply(template_function, axis=1)
            
            # AxBench has much shorter outputs. We follow the original AxBench format.
            if not keep_orig_axbench_format:
                # Apply the template to all output columns
                for column in ["output", "winning_output", "losing_output", "prepend_steered_output", "blend_in_steered_output"]:
                    if column in all_df.columns:
                        apply_output_template(all_df, column)

            # Print sample row data
            print("\n=== Sample Row Data ===")
            sample_row = all_df.iloc[0]
            for column in sample_row.index:
                print(f"\n{column}:")
                print("-" * (len(column) + 1))
                print(f"{sample_row[column]}")
            print("=====================\n")

        return all_df # do nothing, the task will be standard instruction tuning.i



>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)

def load_state(dump_dir):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(dump_dir, STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


<<<<<<< HEAD
def save_state(dump_dir, state):
=======
def save_state(dump_dir, state, concept_metadata):
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    