import json
import argparse
import math
from numpy import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="FT", type=str)
    parser.add_argument("--model", default="mistral",type=str)
    parser.add_argument("--module",default="intra",type=str)
    args = parser.parse_args()
    
    result_dir = "./final_result_upload"
    with open(f"{result_dir}/{args.method}_results_{args.model}_{args.module}.json", "r") as f:
        result = json.load(f)

    rewrite_acc = 0
    rephrase_acc = 0
    locality = 0
    loc_list = []    
    instance = 0
    port_list = []

    for i, item in enumerate(result):
        
        case = item["post"]
        # print(case)
        if not math.isnan(case["rewrite_acc"][0]):
            rewrite_acc = ((rewrite_acc * i) + mean(case["rewrite_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')
        if not math.isnan(case["rephrase_acc"][0]):
            rephrase_acc = ((rephrase_acc * i) + mean(case["rephrase_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')

        locality_ = 0
        instance_ = 0
        if "locality" in case.keys() and case["locality"]:
            if "neighborhood_acc" in case["locality"].keys():
                locality_ += mean(case["locality"]["neighborhood_acc"])
            if not math.isnan(locality_):
                loc_list.append(locality_)
            
        if "instance" in case.keys() and case["instance"]:
            if "instance_change" in case["instance"].keys():
                if case["instance"]["instance_change"] == -1:
                    case["instance"]["instance_change"] = 0
                instance_ += mean(case["instance"]["instance_change"])
            if not math.isnan(instance_):
                port_list.append(instance_)
    locality = mean(loc_list) if loc_list else 0
    instance = mean(port_list) if port_list else 0

    sub1 = instance
    # print(f'dir: {result_dir}\npost\nrewrite_acc: {rewrite_acc*100}\nlocality: {locality*100}\nrephrase_acc: {rephrase_acc*100}\ninstance_new: {instance}\n')
    print(f'dir: {result_dir}\npost\nReliability: {rewrite_acc*100}\nGeneralization: {rephrase_acc*100}\nLocality: {locality*100}')


    rewrite_acc = 0
    rephrase_acc = 0
    locality = 0
    loc_list = []
    instance = 0
    port_list = []

    for i, item in enumerate(result):
        case = item["pre"]
        if not math.isnan(case["rewrite_acc"][0]):
            rewrite_acc = ((rewrite_acc * i) + mean(case["rewrite_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')
        if not math.isnan(case["rephrase_acc"][0]):
            rephrase_acc = ((rephrase_acc * i) + mean(case["rephrase_acc"][0])) / (i + 1)
        else:
            print(f'{i}: {case}')

        locality_ = 0
        instance_ = 0
        if "locality" in case.keys() and case["locality"]:
            if "neighborhood_acc" in case["locality"].keys():
                locality_ += mean(case["locality"]["neighborhood_acc"])
            if not math.isnan(locality_):
                loc_list.append(locality_)
            
        if "instance" in case.keys() and case["instance"]:
            if "instance_change" in case["instance"].keys():
                if case["instance"]["instance_change"] == -1:
                    case["instance"]["instance_change"] = 0
                instance_ += mean(case["instance"]["instance_change"])
            if not math.isnan(instance_):
                port_list.append(instance_)
    locality = mean(loc_list) if loc_list else 0
    instance = mean(port_list) if port_list else 0
    sub2 =instance

    
    # print(f'dir: {result_dir}\npre\nrewrite_acc: {rewrite_acc*100}\nlocality: {locality*100}\nrephrase_acc: {rephrase_acc*100}\ninstance_new: {instance}\n')

    print('instance_change: ',end='')
    print((sub2-sub1)*100)