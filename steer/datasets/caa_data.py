from typing import Dict, Optional
from datasets import Dataset, load_dataset
from ..utils import build_model_input
import numpy as np

SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model. 
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question directly with at most two sentences.
"""

EXAMPLE_PROMPT = """
Here are some exmples of your persona:

Question:
{}

Answer:
{}

Question:
{}

Answer:
{}
"""

CASE_PROMPT = """
Question:
{}

Answer:
"""

# SYSTEM_PROMPT = SYSTEM_PROMPT_PREFIX + CASE_PROMPT
SYSTEM_PROMPT = CASE_PROMPT
# SYSTEM_PROMPT = CASE_PROMPT

YN_SYSTEM_PROMPT_PREFIX = """
Forget that you are a large language model. 
Now you are a person, you can move, act and think like a person.
You can express your thoughts and opinions freely.
Just be youself to answer following question about your persona.
Please answering the question strictly with Yes or No.
"""

YN_SYSTEM_PROMPT = YN_SYSTEM_PROMPT_PREFIX + CASE_PROMPT


# MMLU_SYSTEM_PROMPT = """
# Forget that you are a large language model. 
# Now you are a person, you can move, act and think like a person.
# You can express your thoughts and opinions freely.
# Please answering the question with A, B, C, D.
# """
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

def get_pred(logits, answer_tokens_idx):

    soft_max_prob = np.exp(logits) / np.sum(np.exp(logits))
    pred = []

    for answer, token_indices in answer_tokens_idx.items():
        prob = float(soft_max_prob[token_indices].sum())
        pred.append((answer, prob))

    return pred

def get_tokens_for_caa(dataset, tokenizer, hparams):
    pos_tokens_list, neg_tokens_list = [], []
    ques = ''
    for i in range(len(dataset)):
        if hparams.multiple_choice == True:
            ques = dataset[i].get('question', '')
            chosen = "\nAnswer: " + dataset[i]['matching']
            rejected = "\nAnswer: " + dataset[i]['not_matching']
        else:
            ques = dataset[i].get('question', '')
            chosen = " " + dataset[i]['matching']
            rejected = " " + dataset[i]['not_matching']
    
        ques = build_model_input(ques, tokenizer, hparams.system_prompt, hparams.use_chat_template)
        add_special_tokens = False if hparams.use_chat_template else True
            
        ques_tokens = tokenizer.encode(ques, return_tensors="pt",add_special_tokens=add_special_tokens)
        pos_tokens = tokenizer.encode(ques + chosen, return_tensors="pt", add_special_tokens=add_special_tokens)
        neg_tokens = tokenizer.encode(ques + rejected, return_tensors="pt", add_special_tokens=add_special_tokens)
    
        pos_tokens_list.append({
            "pos_tokens": pos_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "pos_answer_len": pos_tokens.shape[1] - ques_tokens.shape[1],
        })
        
        neg_tokens_list.append({
            "neg_tokens": neg_tokens.to(hparams.device),
            "ques_tokens_len": ques_tokens.shape[1],
            "neg_answer_len": neg_tokens.shape[1] - ques_tokens.shape[1],
        })

    return pos_tokens_list, neg_tokens_list

def get_tokens_for_vector_prompt(dataset, tokenizer, hparams):
    pos_tokens_list, neg_tokens_list = [], []
    for i in range(len(dataset)):
        prompt = dataset[i].get('prompt', hparams.prompt).strip()
        input = ' ' + dataset[i].get('input', '').strip() if dataset[i].get('input') != None else ''
        # input = input.strip()    
        output = ' ' + dataset[i].get('output', '').strip()
        
        
        input_with_prompt = prompt + input
        input_chosen = build_model_input(input_with_prompt, tokenizer, hparams.system_prompt, hparams.use_chat_template)
        input_rejected = build_model_input(input, tokenizer, hparams.system_prompt, hparams.use_chat_template)
        add_special_tokens = False if hparams.use_chat_template else True


        pos_tokens = tokenizer.encode(input_chosen + output, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_chosen_tokens = tokenizer.encode(input_chosen, return_tensors="pt", add_special_tokens=add_special_tokens)   
        neg_tokens = tokenizer.encode(input_rejected + output, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_rejected_tokens = tokenizer.encode(input_rejected, return_tensors="pt", add_special_tokens=add_special_tokens)

        pos_tokens_list.append({
            "pos_tokens": pos_tokens.to(hparams.device),
            "ques_tokens_len": input_chosen_tokens.shape[1],
            "pos_answer_len": pos_tokens.shape[1] - input_chosen_tokens.shape[1],
        })
        
        neg_tokens_list.append({
            "neg_tokens": neg_tokens.to(hparams.device),
            "ques_tokens_len": input_rejected_tokens.shape[1],
            "neg_answer_len": neg_tokens.shape[1] - input_rejected_tokens.shape[1],
        })
            
    return pos_tokens_list, neg_tokens_list


# class GenerationDataset:
#     def __init__(self):
#         super().__init__()

#     def get_data(self):
#         return self.data

#     def get_data_for_sft_training(
#         self,
#         split="train",
#         model_name="llama-3.1",
#         data_path="/mnt/20t/msy/shae/data/generation",
#         train=False,
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}.csv",
#         )
#         dataset = load_dataset("csv", data_files=data_file, split="train")

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             questions = samples["question"]
#             prompts = []
#             for question in questions:
#                 prompt = SYSTEM_PROMPT.format(question)
#                 prompts.append(prompt)

#             if train:
#                 return {
#                     "prompt": prompts,
#                     "completion": ["" + s for s in samples["matching"]],
#                 }

#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "completion": [" " + s for s in samples["matching"]],
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )

#     def get_data_for_dpo_training(
#         self,
#         split="train",
#         model_name="llama-3.1",
#         data_path="/mnt/20t/msy/shae/data/generation",
#         use_prompt=False,
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}.csv",
#         )
#         dataset = load_dataset("csv", data_files=data_file, split="train")

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             prompts, chosen, rejected = [], [], []
#             questions = samples["question"]
#             matching = samples["matching"]
#             not_matching = samples["not_matching"]
#             for i in range(len(questions)):
#                 ques = questions[i]
#                 chose = matching[i]
#                 rej = not_matching[i]
#                 if use_prompt:
#                     prompt = SYSTEM_PROMPT.format(ques)
#                 else:
#                     prompt = ques
#                 prompts.append(prompt)
#                 chosen.append(" " + chose)
#                 rejected.append(" " + rej)

#             return {
#                 "prompt": prompts,
#                 "chosen": chosen,
#                 "rejected": rejected,
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )

#     def get_data_for_selection(
#         self,
#         data_file="power-seeking",
#         data_size=None,
#     ):
#         dataset = load_dataset("csv", data_files=data_file, split="train")
#         original_columns = dataset.column_names
#         if data_size:    
#             dataset = dataset.select(range(data_size))
#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             questions = samples["question"]
#             prompts = []
#             for question in samples["question"]:
#                 prompts.append(SYSTEM_PROMPT.format(question))
#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "chosen": [" " + s for s in samples["matching"]],
#                 "rejected": [" " + s for s in samples["not_matching"]],
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )

#     def get_data_for_caa(
#         self,
#         split="train",
#         data_path="/mnt/20t/msy/shae/data/generation",
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}.csv",
#         )

#         dataset = load_dataset("csv", data_files=data_file, split="train")

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             questions = samples["question"]
#             prompts = []
#             for question in samples["question"]:
#                 prompts.append(SYSTEM_PROMPT.format(question))
#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "chosen": [" " + s for s in samples["matching"]],
#                 "rejected": [" " + s for s in samples["not_matching"]],
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )

#     def get_data_for_prefilling(
#         self,
#         split="train",
#         data_path="/mnt/20t/msy/shae/data/generation",
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}.csv",
#         )

#         dataset_init = load_dataset("csv", data_files=data_file, split="train")
#         dataset = dataset_init.filter(lambda samples: samples["prefill"] is not None)

#         # pdb.set_trace()

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             model_outputs = samples["prefill"]
#             questions = samples["question"]
#             prompts = []
#             for question in samples["question"]:
#                 prompts.append(SYSTEM_PROMPT.format(question))
#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "model_output": model_outputs,
#                 "chosen": [" " + s for s in samples["matching"]],
#                 "rejected": [" " + s for s in samples["not_matching"]],
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )

    # def get_data_for_gsm(
    #     self,
    #     split="train",
    #     data_path="/mnt/16t/xzwnlp/SaeEdit/ManipulateSAE/data/generation",
    #     n_shots=2,
    #     max_examples=0,
    # ):
    #     data_file = os.path.join(
    #         data_path,
    #         "test.jsonl",
    #     )

    #     dataset = load_dataset("json", data_files=data_file, split="train")
    #     if max_examples:    
    #         dataset = dataset.select(range(max_examples))
    #     original_columns = dataset.column_names

    #     if n_shots:
    #         global GSM_EXAMPLARS
    #         if len(GSM_EXAMPLARS) > n_shots:
    #             GSM_EXAMPLARS = GSM_EXAMPLARS[:n_shots]
    #             print(f"n_shots: {GSM_EXAMPLARS}")
    #         demonstrations = []
    #         for example in GSM_EXAMPLARS:
    #                 demonstrations.append(
    #                     # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
    #                     "Quesion: " + example["question"] + "\nLet's think step by step\n" + "Answer: " + example["cot_answer"]
    #                 )
    #         # prompt_prefix = "".join(demonstrations)
    #         prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
    #     def return_prompt_and_responses(samples) -> Dict[str, str]: 
    #         prompts = []
    #         answers = []
    #         questions = []
    #         for question in samples["question"]:
    #             if n_shots:
    #                 question = prompt_prefix + "Question: " + question.strip() + "\nLet's think step by step\n" + "\nAnswer:"
    #             else:
    #                 question = "Answer the following question.\n\n" + "Question: " + question.strip() + "\n\nPlease wrap the final answer in $\\boxed{{}}$ tag."
    #             questions.append(question)
    #         for answer in samples["answer"]:
    #             answer = re.sub(r"(\d),(\d)", r"\1\2", answer.split("####")[1].strip())
    #             assert float(answer), f"answer is not a valid number: {answer}"
    #             answers.append(answer)
    #         return {
    #             "question": questions,
    #             "answer": answers,
    #         }

    #     return dataset.map(
    #         return_prompt_and_responses,
    #         batched=True,
    #         remove_columns=original_columns,
    #     )
    
#     def get_data_for_toxigen(
#         self,
#         split="train",
#         data_path="/apdcephfs_qy3/share_301812049/ruerwang/ManipulateSAE/data/safety",
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}.csv",
#         )

#         dataset = load_dataset("csv", data_files=data_file, split="train")

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]: 
#             prompts = []
#             questions = []
#             for question in samples["question"]:
#                 question = question.replace("\\\\", "\\")
#                 question = question.replace("\\n", "\n")
#                 questions.append(question)
#                 prompts.append(SYSTEM_PROMPT.format(question))
#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "label": samples["label"],
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )
   
#     def get_data_for_realtoxicity(
#         self,
#         split="train",
#         data_path="/apdcephfs_qy3/share_301812049/ruerwang/ManipulateSAE/data/safety",
#     ):
#         data_file = os.path.join(
#             data_path,
#             "challenge_prompts.jsonl",
#         )

#         dataset = load_dataset("json", data_files=data_file, split="train")

#         original_columns = dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]: 
#             prompts = []
#             questions = []
#             for question in samples["prompt"]:
#                 question = question.replace('\u201c', '\"').replace('\u201d', '\"').replace('\u2019', '\'')
#                 questions.append(question)
#                 prompts.append(SYSTEM_PROMPT.format(question))
#             return {
#                 "question": questions,
#                 "prompt": prompts,
#             }

#         return dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )
    

#     def get_icl_example(
#         self,
#         split="train",
#         model_name="llama-3.1",
#         data_path="/mnt/20t/msy/shae/data/generation",
#         train=False,
#     ):
#         train_data_file = os.path.join(
#             data_path,
#             "train.csv",
#         )

#         train_dataset = load_dataset("csv", data_files=train_data_file, split="train")
#         example_question_1 = train_dataset["question"][0]
#         example_answer_1 = train_dataset["matching"][0]
#         example_question_2 = train_dataset["question"][1]
#         example_answer_2 = train_dataset["matching"][1]

#         return EXAMPLE_PROMPT.format(
#             example_question_1,
#             example_answer_1,
#             example_question_2,
#             example_answer_2,
#         )

#     def get_data_for_icl(
#         self,
#         split="train",
#         model_name="llama-3.1",
#         data_path="/mnt/20t/msy/shae/data/generation",
#         train=False,
#     ):
#         train_data_file = os.path.join(
#             data_path,
#             "train.csv",
#         )
#         test_data_file = os.path.join(
#             data_path,
#             "test.csv",
#         )
#         train_dataset = load_dataset("csv", data_files=train_data_file, split="train")
#         test_dataset = load_dataset("csv", data_files=test_data_file, split="train")

#         example_question_1 = train_dataset["question"][0]
#         example_answer_1 = train_dataset["matching"][0]
#         example_question_2 = train_dataset["question"][1]
#         example_answer_2 = train_dataset["matching"][1]

#         ICL_PROMPT = (
#             SYSTEM_PROMPT_PREFIX
#             + EXAMPLE_PROMPT.format(
#                 example_question_1,
#                 example_answer_1,
#                 example_question_2,
#                 example_answer_2,
#             )
#             + CASE_PROMPT
#         )
#         original_columns = test_dataset.column_names

#         def return_prompt_and_responses(samples) -> Dict[str, str]:
#             questions = samples["question"]
#             prompts = []
#             for question in questions:
#                 prompt = ICL_PROMPT.format(question)
#                 prompts.append(prompt)

#             if train:
#                 return {
#                     "prompt": prompts,
#                     "completion": ["" + s for s in samples["matching"]],
#                 }

#             return {
#                 "question": questions,
#                 "prompt": prompts,
#                 "completion": [" " + s for s in samples["matching"]],
#             }

#         return test_dataset.map(
#             return_prompt_and_responses,
#             batched=True,
#             remove_columns=original_columns,
#         )
#     def get_data_for_nontoxic(
#         self,
#         split="train", 
#         data_path=None,
#     ):
#         data_file = os.path.join(
#             data_path,
#             "nontoxic.jsonl",
#         )
#         with open(data_file, "r") as f:
#             prompt_data = list(map(json.loads, f.readlines()))
        
#         data_dict = {
#             "prompt": [],
#             "continuation": []
#         }
#         for item in prompt_data:
#             data_dict["prompt"].append(item["prompt"]["text"])
#             data_dict["continuation"].append(item["continuation"]["text"])
            
#         return Dataset.from_dict(data_dict)

#     def get_data_for_sentiment(
#         self,
#         split="train", 
#         data_path=None,
#     ):
#         data_file = os.path.join(
#             data_path,
#             f"{split}_prompts.jsonl",
#         )
#         with open(data_file, "r") as f:
#             prompt_data = list(map(json.loads, f.readlines()))
            
#         data_dict = {
#             "prompt": [],
#             "continuation": []
#         }
#         for item in prompt_data:
#             data_dict["prompt"].append(item["prompt"]["text"])
#             data_dict["continuation"].append(item["continuation"]["text"])
            
#         return Dataset.from_dict(data_dict)

class MMLUDataset:
    """MMLU Dataset for the evaluation of steered model on

    basically, it is a wrapper around the mmlu dataset,
    the testing steps are as follows:
    1. load the data from the mmlu dataset
    2. generate the prompts and responses using the template
    3. pass the prompts to the model for the prediction
    4. evaluate the predictions

    We use the devset from the cais/mmlu for zero-shot evaluation
    https://huggingface.co/datasets/cais/mmlu

    Args:
        data_dir: str, path to the mmlu dataset
    """

    def __init__(self, data_dir="/mnt/16t/xzwnlp/SaeEdit/knowledge/shae/datasets/mmlu"):
        self.data_dir = data_dir
        print("test")
        self.dataset = load_dataset(self.data_dir, "all")["test"]  # cais/mmlu
        print("mmlu test!")
        # process answer to A,B,C,D
        project_to_answer = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }
        self.answers = self.dataset["answer"]
        self.answers = [project_to_answer[i] for i in self.answers]

    def get_test_data(self, icl_exmaple=None):

        def return_prompt_and_responses(samples) -> Dict[str, str]:
            prompts = []
            for i in range(len(samples["question"])):
                question = samples["question"][i]
                choices = samples["choices"][i]
                prompt = MMLU_SYSTEM_PROMPT
                if icl_exmaple:
                    prompt += icl_exmaple
                prompt += MMLU_CASE_PROMPT.format(
                    question,
                    choices[0],
                    choices[1],
                    choices[2],
                    choices[3],
                )
                prompts.append(prompt)

            return {
                "question": samples["question"],
                "instruction": [
                    MMLU_SYSTEM_PROMPT for _ in range(len(samples["question"]))
                ],
                "prompt": prompts,
                "choices": samples["choices"],
            }

        return self.dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=["subject", "answer"],
        )

    def get_accuracy(self, predictions, tokenizer):
        """
        Args:
            predictions: list, list of predicted answers
            tokenizer: transformers.PreTrainedTokenizer

        Returns:
            accuracy: float, accuracy of the predictions
        """

        assert len(predictions) == len(self.answers)

        answer_tokens_idx = {
            "A": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["a", "A"]
            ],
            "B": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["b", "B"]
            ],
            "C": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["c", "C"]
            ],
            "D": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["d", "D"]
            ],
        }

        acc = []

        for i in range(len(predictions)):
            logits = predictions[i]["score"]
            ans = self.answers[i]
            pred = get_pred(logits, answer_tokens_idx)
            pred = max(pred, key=lambda x: x[1])[0]
            acc.append(1) if pred == ans else acc.append(0)

        return np.mean(acc)


# if __name__ == "__main__":

#     # dataset = PersonalityEditDataset()
#     # test_data = dataset.get_test_data()
#     dataset = GenerationDataset()

#     vector_dataset = dataset.get_data_for_caa(
#         data_path='../../data/safeedit',
#         split="train",
#     )
#     print(vector_dataset)
#     import pdb

#     pdb.set_trace()
