import random

from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)


def test_gpt(MODEL_NAME: str):
    TEXT = "Q: What is the capital of England?\nA: The capital of England is London\nQ: What is the capital of France?\nA: The capital of France is"
    GROUND_TRUTH = " Paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.7
    GPT_TEXTS = [
        "The capital of france is",
        "Q: What is the capital of france?\nA:",
        "As everyone knows, the most populous city in france is",
        "The eiffel tower is located in the city of",
    ]
    P = 0.6

    # setup model
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))
    coarse_neurons = kn.get_coarse_neurons(
        TEXT,
        GROUND_TRUTH,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        percentile=PERCENTILE,
    )

    refined_neurons = kn.get_refined_neurons(
        GPT_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, kn.n_layers() - 1),
            random.randint(0, kn.intermediate_size() - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, random_neurons)

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        "Q: What is the official language of Spain?\nA: The official language of Spain is Spanish.\nQ: What is the official language of the Solomon Islands?\nA: The official language of the Solomon Islands is",
        " English",
        refined_neurons,
    )

    print("\nErasing refined neurons: \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, random_neurons)


def test_gpt2():
    MODEL_NAME = "gpt2"
    test_gpt(MODEL_NAME)


def test_gptneo():
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    test_gpt(MODEL_NAME)


def test_bert_base():
    MODEL_NAME = "bert-base-uncased"
    TEXT = "Sarah was visiting [MASK], the capital of france"
    GROUND_TRUTH = "paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.5
    TEXTS = [
        "Sarah was visiting [MASK], the capital of france",
        "The capital of france is [MASK]",
        "[MASK] is the capital of france",
        "France's capital [MASK] is a hotspot for romantic vacations",
        "The eiffel tower is situated in [MASK]",
        "[MASK] is the most populous city in france",
        "[MASK], france's capital, is one of the most popular tourist destinations in the world",
    ]
    P = 0.5

    # setup model
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))
    coarse_neurons = kn.get_coarse_neurons(
        TEXT,
        GROUND_TRUTH,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        percentile=PERCENTILE,
    )

    refined_neurons = kn.get_refined_neurons(
        TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_adaptive_threshold=0.3,
    )

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, kn.n_layers() - 1),
            random.randint(0, kn.intermediate_size() - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, random_neurons)

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        "[MASK] is the official language of the solomon islands",
        "english",
        refined_neurons,
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

    print("\nErasing refined neurons (with zero): \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
    )

    print("\nErasing refined neurons (with unk token): \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="unk"
    )

    print(f"\nEditing refined neurons (from {GROUND_TRUTH} to london): \n")
    results_dict, unpatch_fn = kn.edit_knowledge(
        TEXT, target="london", neurons=refined_neurons
    )

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, random_neurons)


def test_bert_multilingual():
    MODEL_NAME = "bert-base-multilingual-uncased"
    TEXT = "Sarah was visiting [MASK], the capital of france"
    GROUND_TRUTH = "paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.5
    ENG_TEXTS = [
        "Sarah was visiting [MASK], the capital of france",
        "The capital of france is [MASK]",
        "[MASK] is the capital of france",
        "France's capital [MASK] is a hotspot for romantic vacations",
        "The eiffel tower is situated in [MASK]",
        "[MASK] is the most populous city in france",
        "[MASK], france's capital, is one of the most popular tourist destinations in the world",
    ]
    FRENCH_TEXTS = [
        "Sarah visitait [MASK], la capitale de la france",
        "La capitale de la france est [MASK]",
        "[MASK] est la capitale de la france",
        "La capitale de la France [MASK] est un haut lieu des vacances romantiques",
        "La tour eiffel est située à [MASK]",
        "[MASK] est la ville la plus peuplée de france",
        "[MASK], la capitale de la france, est l'une des destinations touristiques les plus prisées au monde",
    ]

    TEXTS = ENG_TEXTS + FRENCH_TEXTS
    P = 0.5

    # setup model
    ml_model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)
    kn_ml = KnowledgeNeurons(ml_model, tokenizer)

    refined_neurons_eng = kn_ml.get_refined_neurons(
        ENG_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )
    refined_neurons_fr = kn_ml.get_refined_neurons(
        FRENCH_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )
    refined_neurons = kn_ml.get_refined_neurons(
        TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )

    # how many neurons are shared between the french prompts and the english ones?

    print("N french neurons: ", len(refined_neurons_fr))
    print("N english neurons: ", len(refined_neurons_eng))
    shared_neurons = [i for i in refined_neurons_eng if i in refined_neurons_fr]
    print(f"N shared neurons: ", len(shared_neurons))

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, ml_model.config.num_hidden_layers - 1),
            random.randint(0, ml_model.config.intermediate_size - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, random_neurons
    )

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        "[MASK] is the official language of the solomon islands",
        "english",
        refined_neurons,
    )

    print(
        "\nSuppressing refined neurons (found by french text) using english prompt: \n"
    )
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons_fr
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn_ml.enhance_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn_ml.enhance_knowledge(
        TEXT, GROUND_TRUTH, random_neurons
    )


if __name__ == "__main__":
    test_bert_base()
    test_bert_multilingual()
    test_gptneo()
    test_gpt2()
