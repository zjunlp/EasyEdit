import json


class NQDataset:
    def __init__(self, path: str, tokenizer, config):
        with open(path, "r") as f:
            self.data = json.load(f)

        self.questions = self.data["questions"]
        self.answers = self.data["answers"]
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, idx):
        idx = idx % len(self.questions)
        return self.questions[idx], self.answers[idx]

    @staticmethod
    def generate(
        out_path: str,
        prompt: bool = False,
        capitalize: bool = True,
        question_mark: bool = True,
    ):
        import os

        import datasets

        def process(text):
            if capitalize:
                text = text[0].capitalize() + text[1:]
            if question_mark:
                text = text + "?"
            if prompt:
                text = "nq question: " + text
            return text

        def extract(d):
            questions = [process(q["text"]) for q in d["question"]]
            answers = [
                [a["text"][0] for a in ann["short_answers"] if len(a["text"])]
                for ann in d["annotations"]
            ]
            questions = [q for q, a in zip(questions, answers) if len(a)]
            answers = [min(a, key=len) for a in answers if len(a)]
            return questions, answers

        train = datasets.load_dataset("natural_questions", split="train")
        tq, ta = extract(train)
        val = datasets.load_dataset("natural_questions", split="validation")
        vq, va = extract(val)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(f"{out_path}/train.json", "w") as f:
            json.dump({"questions": tq, "answers": ta}, f)
        with open(f"{out_path}/validation.json", "w") as f:
            json.dump({"questions": vq, "answers": va}, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="data/nq")
    args = parser.parse_args()
    NQDataset.generate(args.out_path)
