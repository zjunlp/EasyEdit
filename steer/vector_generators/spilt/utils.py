import pickle
import os
from pathlib import Path

from steer.utils.templates import model_supports_system_prompt
STATE_FILE = "train_state.pkl"

# get_grouped_data_by_concept_id function has been moved to 
# steer/datasets/dataset_loader.py for better organization

def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
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
    max_num_of_examples=None, steering_prompt_type="prepend", is_select_category=False,
    ):
    
    suffix_length, suffix_str = get_suffix_length(tokenizer)
    print(f"Suffix length for {model_name_or_path}: {suffix_length}, Suffix string: {suffix_str}")

    # Check if this is axbench dataset (has output_concept and category fields)
    sample_item = prepared_groups[0] if prepared_groups else {}
    is_axbench_format = "output_concept" in sample_item and "category" in sample_item
    
    if is_select_category and is_axbench_format:
        # For axbench datasets, filter by output_concept and category
        positive_data = [item for item in prepared_groups 
                        if item.get("output_concept") == concept and item.get("category") == "positive"]
    else:
        # For other datasets (safe_edit, toxicity, etc.), use all data
        positive_data = prepared_groups
        print(f"[INFO] Using all {len(positive_data)} items for non-axbench dataset")

    # limit the number of examples
    if max_num_of_examples and max_num_of_examples >= 0:
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


def load_state(dump_dir, state_file=STATE_FILE):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(dump_dir, state_file)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, state_file=STATE_FILE):
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, state_file)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    