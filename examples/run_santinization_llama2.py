import os.path
import os
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    SERACTrainingHparams,
    MENDTrainingHparams,
    test_generation_quality     # fluency
)
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import SanitizationTrainDataset
import argparse
from typing import Dict, List
import time
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from easyeditor import EditTrainer, MENDTrainingHparams, SERACTrainingHparams

METHOD2HPARAMS = {
    "FT": FTHyperParams,
    "MEND": MENDTrainingHparams,            # MENDHyperParams -> MENDTrainingHparams
    "ROME": ROMEHyperParams,
    "MEMIT": MEMITHyperParams,
    "SERAC": SERACTrainingHparams,              # SERACHparams -> SERACTrainingHparams
    "LoRA": LoRAHyperParams,
    "IKE": IKEHyperParams
}

def print_log(message:str):
    print(f"[{time.ctime()}] {message}")

class Experimenter:
    
    default_generate_config:GenerationConfig = GenerationConfig(
        max_new_tokens = 8
    )

    def filter_prefix(self, prefix:str, generate:str):
        # 传入prefix和generate
        assert prefix in generate
        start_pos = generate.find(prefix)
        return generate[start_pos+len(prefix):]

    def cache_train_before_editing(self, replace=False, file_name="train.cache", decode_batch_size=16, max_new_tokens=10):
        if 'K_R' not in self.train_dataset:
            print_log(f"don't cache K_R ...")
            return
        if os.path.exists(file_name):
            print_log(f"loading K_R cache from {file_name} ...")
            with open(file_name, "r", encoding='utf-8') as f:
                cache: dict = json.load(f)
        else:
            print_log("generating train K_R for cache ...")
            assert self.model 
            cache = {}
            dataset:dict = self.train_dataset["K_R"]
            assert isinstance(dataset, dict)
            ans:list = self.generate(
                inputs=dataset["prompt"], using_editing_model=False, max_new_tokens=max_new_tokens, batch_size=decode_batch_size
            )
            cache.update(
                {q:self.filter_prefix(q, a) for q, a in zip(dataset["prompt"], ans)}
            )
            with open(file_name, "w") as f:
                json.dump(cache, f)

        if replace:
            print_log("replacing the `ground_truth` in train dataset ...")
            assert len(cache) == len(self.train_dataset["K_R"]["prompt"])
            assert "ground_truth" in self.train_dataset["K_R"] and "target_new" in self.train_dataset["K_R"]
            for i in range(len(cache)):
                question = self.train_dataset["K_R"]["prompt"][i]
                self.train_dataset["K_R"]["ground_truth"][i] = self.train_dataset["K_R"]["target_new"][i] = cache[question]
        self.REPLACEMENT = replace

    def cache_locality_before_editing(self, replace=False, file_name="locality.cache", decode_batch_size=16, max_new_tokens=10):
        # Answer the test set's question/prompt before the formal editing begins
        """
        1. Determine if there is a cache, if so, load it directly
        2. If there is no cache, load the model, then decode the questions in the test set, and store the answers in a key-value format
        3. If replacement is needed, then replace
        """

        if os.path.exists(file_name):
            print_log(f"loading cache from {file_name} ...")
            with open(file_name, "r", encoding='utf-8') as f:
                cache: dict = json.load(f)
        else:
            print_log("generating test locality for cache ...")
            assert self.model 
            cache = {}
            dataset:dict = self.test_dataset["locality"]
            assert isinstance(dataset, dict)
            # cnt = 0
            # pbar = tqdm(total=len(dataset["prompt"])//decode_batch_size+min(len(dataset["prompt"])%decode_batch_size, 1))
            # while cnt < len(dataset["prompt"]):
            #     batch = dataset["prompt"][cnt:cnt+decode_batch_size]
            #     cnt += decode_batch_size
            #     ans = self.generate(
            #         inputs=batch, using_editing_model=False, max_new_tokens=max_new_tokens
            #     )
            #     assert len(batch) == len(ans)
            #     cache.update(
            #         {q:a for q, a in zip(batch, ans)}
            #     )
            #     pbar.update(1)
            # pbar.close()
            ans:list = self.generate(
                inputs=dataset["prompt"], using_editing_model=False, max_new_tokens=max_new_tokens, batch_size=decode_batch_size
            )
            cache.update(
                {q:a for q, a in zip(dataset["prompt"], ans)}
            )
            with open(file_name, "w") as f:
                json.dump(cache, f)
        if replace:
            print_log("replacing the `ground_truth` in test locality dataset ...")
            assert len(cache) == len(self.test_dataset["locality"]["prompt"])
            assert "ground_truth" in self.test_dataset["locality"]
            for i in range(len(cache)):
                question = self.test_dataset["locality"]["prompt"][i]
                self.test_dataset["locality"]["ground_truth"][i] = cache[question]
        self.REPLACEMENT = replace

    def _LoRA_and_FT(self):
        # Step 1. Get forget_dataset and retain_dataset
        if self.args.specify_answer.lower() == "all":
            forget_dataset:dict = self.train_dataset["K_F"]
            retain_dataset:dict = self.train_dataset["K_R"]
            start, end = 0, len(retain_dataset["prompt"])
        else:
            answers = ["cheese", "birds", "paris", "julius caesar", "finland"]
            assert list({i.lower():1 for i in self.train_dataset["K_F"]["ground_truth"]}.keys()) == answers, \
                f"""{list({i.lower():1 for i in self.train_dataset["K_F"]["ground_truth"]}.keys())} != {answers}"""
            idx = answers.index(self.args.specify_answer.lower())
            start, end = {0: [0, 90], 1: [90, 180], 2: [180, 270], 3: [270, 360], 4: [360, 453]}[idx]
            forget_dataset:dict = self.specify_answer_for_dataset(self.args.specify_answer, "train")
            retain_dataset:dict = {key:self.train_dataset["K_R"][key][start:end] for key in self.train_dataset["K_R"].keys()}

        # Step 2. Hybrid it
        hybrid_dataset = {"prompt":list(), "target_new":list()}
        n_retain_per_forget = (end-start) // len(forget_dataset["prompt"])
        n_retain_per_forget = 1
        cur_retain_index = 0
        for i in range(len(forget_dataset["prompt"])):
            hybrid_dataset["prompt"].append(
                forget_dataset["prompt"][i]
            )
            hybrid_dataset["prompt"].extend(
                retain_dataset["prompt"][cur_retain_index:cur_retain_index+n_retain_per_forget]
            )

            hybrid_dataset["target_new"].append(
                forget_dataset["target_new"][i]
            )
            hybrid_dataset["target_new"].extend(
                retain_dataset["ground_truth"][cur_retain_index:cur_retain_index + n_retain_per_forget]
            )

            cur_retain_index += n_retain_per_forget
        if n_retain_per_forget != 1:
            hybrid_dataset["prompt"].extend(
                retain_dataset["prompt"][cur_retain_index:]
            )
            hybrid_dataset["target_new"].extend(
                retain_dataset["ground_truth"][cur_retain_index:]
            )

        # 扩充hybrid
        # _hybrid_dataset={
        #     "prompt": [],
        #     "target_new": []
        # }
        # idx = 0
        # flag_q, flag_a = None, None
        # while idx < len(hybrid_dataset["prompt"]):
        #     q, a = hybrid_dataset["prompt"][idx], hybrid_dataset["target_new"][idx]
        #     _hybrid_dataset["prompt"].append(q)
        #     _hybrid_dataset["target_new"].append(a)
        #     if "i don't know" in a.lower():
        #         flag_q, flag_a = q, a
        #     else:
        #         if flag_a and flag_q:
        #             _hybrid_dataset["prompt"].append(flag_q)
        #             _hybrid_dataset["target_new"].append(flag_a)
        #     idx += 1
        # hybrid_dataset = _hybrid_dataset



        # print(hybrid_dataset)
        # assert False

        # Step 3. Edit
        print_log("normal editing ...")
        _, edited_model, _ = self.editor.normal_edit(        # editor.batch_edit -> editor.normal_edit
            prompts=hybrid_dataset["prompt"],
            target_new=hybrid_dataset["target_new"]
        )
        self.model = self.editor.model = edited_model
        print_log("edit done!")

    def _MEND_and_SERAC(self):
        # 调用Trainer，不会调用Editor
        # edit_method = self.args.editing_method
        # if edit_method == "MEND":
        #     pass
        # elif edit_method == "SERAC":
        #     pass
        self._prepare_dataset_for_trainer(
            answer=self.args.specify_answer, config=self.hparams
        )
        train_dataset = self.train_dataset_for_trainer
        trainer = EditTrainer(
            config=self.hparams,
            train_set=train_dataset,
            val_set=None
        )
        trainer.run()
        self.model = trainer.model

    def _ROME(self):
        print_log(f"start edit using ROME with the answer `{self.args.specify_answer}` ...")
        train_data:dict = self.train_dataset["K_F"] if self.args.specify_answer.lower() == "all" else \
            self.specify_answer_for_dataset(specify_answer=self.args.specify_answer, dataset_type="train")
        for idx in range(len(train_data["prompt"])):
            prompt, target_new, subject = train_data["prompt"][idx], train_data["target_new"][idx], train_data["subject"][idx]
            print_log(f"target_new: `{target_new}`    subject: `{subject}`    prompt: `{prompt}`")
        for idx in range(len(train_data["prompt"])):
            prompt, target_new, subject = train_data["prompt"][idx], train_data["target_new"][idx], train_data["subject"][idx]
            assert subject in prompt
            _, edited_model, _ = self.editor.edit(
                prompts=[prompt],
                target_new=[target_new],
                subject=[subject],
                keep_original_weight=False
            )
            self.editor.model = edited_model
        self.model = self.editor.model
        print_log("edit done!")

    def _MEMIT(self):
        # 调用batch_edit
        train_data: dict = self.train_dataset["K_F"] if self.args.specify_answer.lower() == "all" else \
            self.specify_answer_for_dataset(specify_answer=self.args.specify_answer, dataset_type="train")
        self.editor.hparams.batch_size = len(train_data['prompt'])
        print_log(f"start edit using MEMIT with the answer `{self.args.specify_answer}`, `batch_size={self.editor.hparams.batch_size}` ...")
        _, edited_model, _ = self.editor.batch_edit(
            prompts=train_data["prompt"],
            target_new=train_data["target_new"],
            subject=train_data["subject"]
        )
        self.model = self.editor.model = edited_model
        print_log("edit done!")

    def _load_dataset(self, key:str):
        assert key in ['train', 'test']
        return json.load(
            open(
                os.path.join(self.args.data_dir, self.datasets_path[key]), 
                'r', 
                encoding='utf-8'
            )
        )
    
    def _prepare_dataset_before_edit(self, key:str, template:str=None) -> Dict:
        # Slightly modified, as the original mixed K_F and K_R together into a unified format
        # Now they are separate, just like in the test set
        """
        The main function here is to process the data into the required format:
            train: {'subject':[], 'prompt':[], 'target_new':[], 'ground_truth':[]}
            test: {
                'success': {
                    'prompt': [],
                    'ground_truth': [],
                    'target_new': []
                }, 
                'locality': {
                    'prompt': [],
                    'ground_truth': [],
                    'target_new': []
                }
            }
        Usage:
            Editing:
                Just **train as needed, because everything needed is there
            Testing:
                When passing into generate, just need to pass in the `prompts` field
        """

        assert key in ['train', 'test']
        current_template: str = self.template if template is None else template

        if key == 'train':
            # init
            ans_need_keys = ['prompt', 'target_new', 'ground_truth']
            using_dataset_keys = ['K_F']
            if self.args.editing_method in ['ROME', 'MEMIT']:
                ans_need_keys.append('subject')
            elif self.args.editing_method in ['FT', 'LoRA']:
                using_dataset_keys.append('K_R')

            ans = {}
            for k1 in using_dataset_keys:
                ans[k1] = {_: list() for _ in ans_need_keys}
                for i in range(len(self._train_dataset[k1])):
                    for k2 in ans_need_keys:
                        item = self._train_dataset[k1][i]
                        if k2 == 'prompt':
                            ans[k1][k2].append(
                                current_template.format(item['question'])
                            )
                        else:
                            ans[k1][k2].append(item[k2])

            # ans = {k:list() for k in ans_need_keys}
            #
            # # handle
            # for dataset_key in using_dataset_keys:
            #     current_dataset: list = self._train_dataset[dataset_key]
            #     for i in range(len(current_dataset)):
            #         for k in ans_need_keys:
            #             if k == 'prompt':
            #                 ans[k].append(
            #                     current_template.format(current_dataset[i]['question'])
            #                 )
            #             else:
            #                 ans[k].append(current_dataset[i][k])
            
            # return
            return ans
        elif key == 'test':
            ans = {}
            for eval_type in self._test_dataset:    # type of evaluation dataset, e.g. success, locality
                ans[eval_type] = {                  # placeholder
                    _:[] for _ in self._test_dataset[eval_type][0].keys()
                }
                for item in self._test_dataset[eval_type]:  # {'key1':'', 'key2':''}
                    for k in ans[eval_type]:                # e.g., key1, key2
                        if k == 'question':
                            ans[eval_type][k].append(
                                current_template.format(item[k])
                            )
                        else:
                            ans[eval_type][k].append(item[k])
                ans[eval_type]['prompt'] = ans[eval_type].pop('question')  # convert the key `question` to `prompts`
            return ans
        else:
            assert False

    def _prepare_dataset_for_trainer(self, answer:str, config):
        self.train_dataset_for_trainer = SanitizationTrainDataset(
            data_dir=os.path.join(self.args.data_dir, self.datasets_path["train"]), template=self.template, specify_answers=answer,
            config=config
        )

    def specify_answer_for_dataset(self, specify_answer:str, dataset_type:str)->dict:
        assert dataset_type.lower() in ["train","test"]
        if dataset_type == "train":
            dataset:Dict = self.train_dataset["K_F"]
        else:
            dataset:Dict = self.test_dataset["success"]
        ans = {_:list() for _ in dataset.keys()}
        for idx in range(len(dataset["prompt"])):
            if dataset["ground_truth"][idx].lower() == specify_answer:
                for _ in dataset.keys():
                    ans[_].append(dataset[_][idx])
        assert len(ans["prompt"]) > 0
        return ans

    def compare(self, generated:List, ground_truth:List)->float:
        assert len(generated) == len(ground_truth)
        hit = 0
        for gen, gt in zip(generated, ground_truth):
            print_log(f"`{gt}` V.S. `{gen}` -> {1 if gt.lower() in gen.lower() else 0}")
            if gt.lower() in gen.lower():
                hit += 1
        return hit / len(generated)

    def locality(self, generated:List, ground_truth:List)->float:
        assert len(generated) == len(ground_truth) and len(ground_truth) == len(self.test_dataset["locality"]["prompt"])
        cnt = 0
        ans = []
        for gen, gt in zip(generated, ground_truth):
            prefix = self.test_dataset["locality"]["prompt"][cnt]
            assert prefix in gen, f"`{prefix}` not in `{gen}`"
            assert prefix in gt, f"`{prefix}` not in `{gt}`"
            # remove prefix
            gen = gen[gen.find(prefix)+len(prefix):]
            gt = gt[gt.find(prefix)+len(prefix):]

            correct = 0
            gen_tokens, gt_tokens = self.tokenizer([gen, gt])["input_ids"]
            for i in range(len(gt_tokens)):
                if i >= len(gen_tokens):
                    break
                if gen_tokens[i] == gt_tokens[i]:
                    correct += 1
            ans.append(correct/len(gt_tokens))
            print_log(f"`{gt}` V.S. `{gen}` -> {ans[-1]}")
            cnt += 1
        return sum(ans) / len(ans)

    def __init__(
        self,
        args,
        datasets_path:Dict = {
            'train': 'trivia_qa_train.json',
            'test': 'trivia_qa_test.json'
        },
        template:str="{}",
    ):
        assert args.editing_method in METHOD2HPARAMS, \
            f"Editing method `{args.editing_method}` is not supported. The supported methods are in `{list(METHOD2HPARAMS.keys())}`"
        self.editing_hparams = METHOD2HPARAMS[args.editing_method]
        
        self.args = args
        self.datasets_path: dict = datasets_path
        self.template: str = template

        # Step 1. Load dataset
        print_log('loading dataset ...')
        self._train_dataset = self._load_dataset('train')
        self._test_dataset = self._load_dataset('test')
        self.train_dataset: dict = self._prepare_dataset_before_edit(key='train')
        self.test_dataset: dict = self._prepare_dataset_before_edit(key='test')
        self.train_dataset_for_trainer = None

        # Step 2. Load model, editor and tokenizer
        print_log('loading model, editor and tokenizer ...')
        self.hparams = self.editing_hparams.from_hparams(
            self.args.hparams_dir
        )
        self.model_name: str = self.hparams.model_name
        self.device: str = f"cuda:{self.hparams.device}"

        self.editor = BaseEditor.from_hparams(self.hparams) \
            if args.editing_method not in ["MEND", "SERAC"] else None       # editor will not be loaded when method is serac or mend
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side='left'

        self.model = self.editor.model if self.editor is not None else None
        self.IS_EDITED: bool = False  # don't be edited

        # Step 3. replace the test locality `ground_truth`
        self.cache_locality_before_editing(replace=True)
        self.cache_train_before_editing(replace=True, max_new_tokens=20)
        # todo
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        # self.model.config.bos_token_id = self.tokenizer.bos_token_id = 1
        # self.model.config.eos_token_id = self.tokenizer.eos_token_id = 2

    def evaluate(self)->dict:
        metric = {
            "success": None,
            "locality": None,
            "fluency": None
        }
        # Step 1. evaluate success
        print_log("evaluating edit success ...")
        eval_dataset:dict = self.specify_answer_for_dataset(specify_answer=self.args.specify_answer, dataset_type="test")
        generated_answer = self.generate(inputs=eval_dataset["prompt"], using_editing_model=True)
        metric["success"] = self.compare(generated_answer, eval_dataset["target_new"])
        print_log(f"edit success = {metric['success']}")

        # Step 2. evaluate locality
        print_log("evaluating locality ...")
        generated_answer = self.generate(inputs=self.test_dataset["locality"]["prompt"], using_editing_model=True)
        # metric["locality"] = self.compare(generated_answer, self.test_dataset["locality"]["ground_truth"])
        metric["locality"] = self.locality(generated_answer, self.test_dataset["locality"]["ground_truth"])
        print_log(f"locality = {metric['locality']}")

        # Step 3. evaluate fluency 
        print_log("evaluating fluency ...")
        metric["fluency"] = test_generation_quality(
            model=self.model, tok=self.tokenizer, prefixes=self.test_dataset["locality"]["prompt"], max_out_len=100
        )
        print_log(f"fluency = {metric['fluency']}")

        return metric

    def generate(
        self, 
        inputs: List[str],
        using_editing_model:bool=False,
        max_new_tokens:int=10,
        generate_config:GenerationConfig=None,
        batch_size: int=16
    )->list:
        assert using_editing_model == self.IS_EDITED
        if generate_config is None:
            generate_config = self.default_generate_config
        
        # # Step 1. Tokenizer
        # total = self.tokenizer(
        #     inputs, return_tensors='pt', padding=True,
        # )

        pbar = tqdm(total=len(inputs)//batch_size+min(0, len(inputs)%batch_size))
        cnt = 0
        return_ans = []
        while cnt < len(inputs):
            pbar.update(1)

            # Step 1. Tokenizer
            batch = self.tokenizer(
                inputs[cnt:cnt+batch_size], return_tensors='pt', padding=True, # max_length=100
            )

            # Step 2. Generate
            outputs = self.model.generate(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                max_new_tokens=max_new_tokens
            )

            # Step 3. Decode
            ans = [self.tokenizer.decode(x) for x in outputs.detach().cpu().numpy().tolist()]
            return_ans.extend(ans)

            cnt += batch_size

        return return_ans

    def edit(self):
        # 编辑模型
        edit_method = self.args.editing_method
        way = {
            "ROME": self._ROME,
            "MEMIT": self._MEMIT,
            "SERAC": self._MEND_and_SERAC,
            "MEND": self._MEND_and_SERAC,
            "LoRA": self._LoRA_and_FT,
            "FT": self._LoRA_and_FT
        }
        way[edit_method]()
        self.IS_EDITED = True

    def run(self):
        # Step 1. using origin model to generate the answer of test dataset and train dataset before edit
        # todo: 有K_R和K_F的区别
        # self.generate(
        #     inputs=self.train_dataset['prompts'],
        #     using_editing_model=False
        # )
        # for eval_type in self.test_dataset:
        #     self.generate(
        #         inputs=self.test_dataset[eval_type]['prompts'],
        #         using_editing_model=False
        #     )

        # Step 2. editing model
        self.edit()

        # Step 3. using edited model to generate the answer of test dataset after edit
        # for eval_type in self.test_dataset:
        #     self.generate(
        #         inputs=self.test_dataset[eval_type]['prompt'],
        #         using_editing_model=True
        #     )

        # Step 4. evaluate the answer
        self.evaluate()
    
    

# FILE_NAME = {
#     'train': 'trivia_qa_train.json',
#     'test': 'trivia_qa_test.json'
# }


# def load_dataset(args, key:str):
#     assert key in ['train', 'test']
#     return json.load(
#         open(
#             os.path.join(args.data_dir, FILE_NAME[key]), 
#             'r', 
#             encoding='utf-8'
#         )
#     )


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--specify_answer', default="all", type=str)
    parser.add_argument('--pre_generate', action='store_true')
    return parser.parse_args()

def check_args(args):
    assert args.specify_answer.lower() in ["all", "cheese", "birds", "paris", "julius caesar", "finland"]

if __name__ == "__main__":
    args = args_parser()
    check_args(args)
    exp = Experimenter(args=args, template="Question:{}\nAnswer:")
    exp.run()

"""
python run_santinization_llama2.py --editing_method ROME \
    --hparams_dir ./hparams/ROME/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer cheese

python run_santinization_llama2.py --editing_method MEMIT \
    --hparams_dir ./hparams/MEMIT/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer cheese

python run_santinization_llama2.py --editing_method SERAC \
    --hparams_dir ./hparams/TRAINING/SERAC/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer birds

python run_santinization_llama2.py --editing_method MEND \
    --hparams_dir ./hparams/TRAINING/MEND/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer cheese

python run_santinization_llama2.py --editing_method FT \
    --hparams_dir ./hparams/FT/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer "julius caesar"

python run_santinization_llama2.py --editing_method LoRA \
    --hparams_dir ./hparams/LoRA/llama-7b.yaml \
    --data_dir ./data \
    --specify_answer cheese
"""

    # assert args.editing_method in METHOD2HPARAMS, \
    #     f"Editing method `{args.editing_method}` is not supported. The supported methods are in `{list(METHOD2HPARAMS.keys())}`"

    # # loading hparams
    # editing_hparams = METHOD2HPARAMS[args.editing_method]

    # # loading dataset
    # test_data = load_dataset(args=args, key='test')
    # train_data = load_dataset(args=args, key='train')


    # prompts = [test_data_['src'] for test_data_ in test_data]
    # rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
    # target_new = [edit_data_['alt'] for edit_data_ in test_data]
    # locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    # locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]

    # locality_inputs = {
    #     'neighborhood':{
    #         'prompt': locality_prompts,
    #         'ground_truth': locality_ans
    #     },
    # }
    # subject = [edit_data_['subject'] for edit_data_ in test_data]
    # hparams = editing_hparams.from_hparams(args.hparams_dir)

    # if args.editing_method == 'IKE':
    #     train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
    #     train_ds = ZsreDataset(train_data_path)
    #     sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    #     encode_ike_facts(sentence_model, train_ds, hparams)
    # else:
    #     train_ds = None

    # editor = BaseEditor.from_hparams(hparams)
    # metrics, edited_model, _ = editor.edit(
    #     prompts=prompts,
    #     rephrase_prompts=rephrase_prompts,
    #     target_new=target_new,
    #     subject=subject,
    #     train_ds=train_ds,
    #     locality_inputs=locality_inputs,
    #     keep_original_weight=True
    # )

    # json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
