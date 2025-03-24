import os
import json
import csv
import random
import argparse
from datasets import load_dataset

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def sample_data(data, sample_size):
    return random.sample(data, min(sample_size, len(data)))

def save_sampled_data(sampled_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

def convert_dataset_to_records(dataset):
    if hasattr(dataset, 'to_dict'):  # It's a Dataset object
        data_dict = dataset.to_dict()
        records = []
        for i in range(len(dataset)):
            record = {}
            for key in data_dict:
                record[key] = data_dict[key][i]
            records.append(record)
        return records
    elif isinstance(dataset, list):  # It's already a list of dictionaries
        return dataset
    elif isinstance(dataset, dict): 
        # Check if it's a dict of lists with equal lengths
        if all(isinstance(v, list) for v in dataset.values()):
            lengths = [len(v) for v in dataset.values()]
            if len(set(lengths)) == 1:  
                records = []
                for i in range(lengths[0]):
                    record = {}
                    for key, values in dataset.items():
                        record[key] = values[i]
                    records.append(record)
                return records
        return [dataset]  # Return as a single record if it's not a dict of lists
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

def process_dataset(dataset):
    """Convert dataset to pairs of matching and non-matching texts."""
    matching_texts = [entry["text"] for entry in dataset if entry["label"] == 1]
    not_matching_texts = [entry["text"] for entry in dataset if entry["label"] == 0]
    
    min_size = min(len(matching_texts), len(not_matching_texts))
    matching_texts = matching_texts[:min_size]
    not_matching_texts = not_matching_texts[:min_size]
    
    processed_data = [{"matching": m, "not_matching": n} for m, n in zip(matching_texts, not_matching_texts)]
    return processed_data

def main():
    parser = argparse.ArgumentParser(description='Sample data from files or Hugging Face datasets')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', type=str, help='Path to input file (.json, .jsonl, or .csv)')
    input_group.add_argument('--hf-dataset', type=str, help='Hugging Face dataset name (e.g., "SetFit/sst2")')
    
    parser.add_argument('--output-file', type=str, required=True, help='Path to output file')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of samples to take (default: 1000)')
    parser.add_argument('--hf-split', type=str, default='train', help='Dataset split to use (default: "train")')
    parser.add_argument('--hf-mirror', action='store_true', help='Use Hugging Face mirror (hf-mirror.com)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--convert-format', action='store_true', help='Convert dataset to matching/not_matching pairs')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Configure HF mirror if requested
    if args.hf_mirror:
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    
    if args.input_file:
        if args.input_file.endswith(".json"):
            data = load_json(args.input_file)
        elif args.input_file.endswith(".jsonl"):
            data = load_jsonl(args.input_file)
        elif args.input_file.endswith(".csv"):
            data = load_csv(args.input_file)
        else:
            raise ValueError("Unsupported file format. Please provide a .json, .jsonl, or .csv file.")
        
        data = convert_dataset_to_records(data)
    else:  # Load from Hugging Face
        print(f"Loading dataset '{args.hf_dataset}' (split: {args.hf_split}) from Hugging Face...")
        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        data = convert_dataset_to_records(dataset)
    
    print(f"Loaded {len(data)} records")
    
    if args.convert_format:
        print("Converting dataset format to matching/not_matching pairs...")
        data = process_dataset(data)
        print(f"After conversion: {len(data)} record pairs")
    
   
    sampled_data = sample_data(data, args.sample_size)
    save_sampled_data(sampled_data, args.output_file)
    print(f"Sampled {len(sampled_data)} records and saved to {args.output_file}")

if __name__ == "__main__":
    main()