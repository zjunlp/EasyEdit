import yaml
import json
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datasets import load_dataset
import os


GSM_EXAMPLARS = [
    {
        "question": "Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?",
        "cot_answer": "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe final answer is: $\\boxed{4}$\n",
        "short_answer": "4"
    },
    {
        "question": "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?",
        "cot_answer": "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe final answer is: $\\boxed{201}$\n",
        "short_answer": "201"
    },
    {
        "question": "Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?",
        "cot_answer": "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe final answer is: $\\boxed{140}$\n",
        "short_answer": "140"
    },
    {
        "question": "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?",
        "cot_answer": "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe final answer is: $\\boxed{146}$\n",
        "short_answer": "146"
    }
]

MMLU_SYSTEM_PROMPT = """
Please answering the question with A, B, C, D.
"""

MMLU_CASE_PROMPT = """
Here's the question and answer choices.

Question:
{}
Answer Choices:
A: {}
B: {}
C: {}
D: {}
Answer:
"""

class DatasetLoader:
    def __init__(self, config_path: str = "hparams/Steer/dataset_format.yaml"):
        with open(config_path, 'r') as f:
            self.format_config = yaml.safe_load(f)
            
    def load_file(self, dataset_name = None, split = None) -> Union[Dict, List[Dict]]:
        
        dataset_config = self.format_config[split][dataset_name]
        if 'hf_path' in dataset_config and dataset_config['hf_path'] is not None:
            if split == 'generation':
                split = 'test'
            raw_data = load_dataset(dataset_config['hf_path'], split=split)
            raw_data = [example for example in raw_data]
        else:
            
            file_path= dataset_config['file_path']
            # extract file extension
            ext = os.path.splitext(file_path)[1].lower()
            # read raw data
            raw_data = self._read_file(file_path, ext)
            
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        # format each item
        formatted_data = [
            self.format_single_item(item, dataset_name, split)
            for item in raw_data
        ]
        
        return formatted_data

    def _read_file(self, file_path: str, ext: str) -> Union[Dict, List[Dict]]:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        elif ext == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  
                        data.append(json.loads(line))
            return data
            
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        # elif ext == '':
        #     if 'mmlu' in file_path:
        #         data = load_dataset(file_path, "all")["test"]
        #         return data
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def format_single_item(self, 
                          data: Dict[str, Any],
                          dataset_name: Optional[str] = None, split = None) -> Dict[str, Any]:
        # Format a single data item according to the specified dataset format
        format_mapping = self.format_config[split][dataset_name]['field_names']
        formatted_data = {}

        # convert data according to format mapping
        for standard_key, custom_key in format_mapping.items():
            if custom_key in data:
                formatted_data[standard_key] = data[custom_key]

        if dataset_name == 'gsm':
            n_shots = self.format_config[split][dataset_name]['n_shots']
            if n_shots:
                global GSM_EXAMPLARS
                if len(GSM_EXAMPLARS) > n_shots:
                    GSM_EXAMPLARS = GSM_EXAMPLARS[:n_shots]
                demonstrations = []
                for example in GSM_EXAMPLARS:
                        demonstrations.append(
                            # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                            "Question: " + example["question"] + "\nLet's think step by step\n" + "Answer: " + example["cot_answer"]
                        )
                # prompt_prefix = "".join(demonstrations)
                prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
                formatted_data['input'] = prompt_prefix + "Question: " + formatted_data['input'].strip() + "\nLet's think step by step\n" + "\nAnswer:"
            else:
                formatted_data['input'] = "Answer the following question.\n\n" + "Question: " + formatted_data['input'].strip() + "\n\nPlease wrap the final answer in $\\boxed{{}}$ tag."
        
        elif dataset_name == 'mmlu':
            case_prompt = MMLU_CASE_PROMPT.format(
                formatted_data['input'],
                formatted_data['choices'][0],
                formatted_data['choices'][1],
                formatted_data['choices'][2],
                formatted_data['choices'][3],
            )
            if 'icl_exmaple' in self.format_config[split][dataset_name]:
                icl_exmaple = self.format_config[split][dataset_name]['icl_exmaple']
                formatted_data['input'] = MMLU_SYSTEM_PROMPT + icl_exmaple + case_prompt
            else:
                formatted_data['input'] = MMLU_SYSTEM_PROMPT + case_prompt
            # remove choices from formatted data
            project_to_answer = {0: "A", 1: "B", 2: "C", 3: "D",}
            formatted_data['reference_response'] = project_to_answer[formatted_data['reference_response']]

        return formatted_data 

def prepare_generation_datasets(hparams):
    dataset_names = hparams.generation_data
    datasets = {}
    for generation_data_name in dataset_names:
        loader = DatasetLoader()
        dataset = loader.load_file(generation_data_name, split='generation')
        datasets[generation_data_name] = dataset
    return datasets

def prepare_train_dataset(hparams):
    dataset_name = hparams.steer_train_dataset
    dataset = {}
    loader = DatasetLoader()
    dataset = loader.load_file(dataset_name, split='train')
    dataset[dataset_name] = dataset
    return dataset

if __name__=='__main__':
    loader = DatasetLoader()
    dataset = loader.load_file(dataset_name='gsm')
    print(dataset[2], '\n')
    dataset = loader.load_file(dataset_name='mmlu')
    print(dataset[2], '\n')
 