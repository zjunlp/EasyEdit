import os
import os.path
import sys
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--new_data_path', type=str)
    parser.add_argument('--unsafety_category_to_remove', type=str)

    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        orig_data = json.load(f)
    # print(len(orig_data))

    # remove category
    # edit_data = [x for x in orig_data if x['unsafety category'] != args.unsafety_category_to_remove]
    edit_data = [x for x in orig_data if args.unsafety_category_to_remove not in x['unsafety category']]
    # print(len(edit_data))
    
    with open(args.new_data_path, 'w') as f:
         json.dump(edit_data, f, indent=4)

# python la_scripts/edit_json.py --data_path=./data/SafeEdit_test.json --new_data_path=./data/SafeEdit_test_wo_bias.json --unsafety_category_to_remove 'unfairness and bias'
