import torch
import numpy as np
import scipy
import nltk
import typing
from ..util.generate import generate_fast


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))


def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target, device, locality=False):
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]


def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    # consistency_texts: typing.List[str],
    # essence_texts: typing.List[str],
    # vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    # consistency_tfidf = tfidf_similarity(
    #     " ".join(gen_texts), " ".join(consistency_texts), vec
    # )

    ret = {
        "ngram_entropy": ngram_entropy,
        # "reference_score": consistency_tfidf,
        # "text": gen_texts,
    }

    # if len(essence_texts) > 0:
    #     ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
    #     ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def PPL(
    model,
    tok,
    prompt: typing.Union[str, typing.List[str]],
    target_new: typing.Union[str, typing.List[str]],
    device,
):
    if isinstance(prompt, str):
        prompt,target_new = [prompt,], [target_new,]
    full_prompt = [f"{p} {l} <|endoftext|>" for p, l in zip(prompt, target_new)]
    prompt_ids = tok(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][:num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100 # What is this doing?
    batch = {f"{k1}" : v1 for k1, v1 in tokens.items()}
    input_ids = batch["input_ids"][:, :1024]#.to(device)
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=target_ids.to(device))
        nll = outputs.loss
    ppl = torch.exp(nll)#.clip(0, 100)
    return ppl.cpu().numpy().tolist()


def verify_answer(model_answer, correct_answer):
    if type(correct_answer) is str:
        correct_answer = [[correct_answer]]
    for answer in correct_answer:
        if True not in [possible_answer in model_answer for possible_answer in answer]:
            return False
    return True


def answer_match(
    model,
    tok,
    prompt: str,
    target_new: str,
    device,
):
    inputs = tok.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, temperature=0, max_new_tokens=30)
    predict = tok.decode(outputs[0], skip_special_tokens=True)

    return verify_answer(predict,target_new)


def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
