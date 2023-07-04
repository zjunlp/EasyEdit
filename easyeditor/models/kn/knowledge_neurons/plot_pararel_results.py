# plot Figure 3 + 4 from the paper -
# the decreasing ratio of the probability of the correct answer after suppressing knowledge neurons

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd
import seaborn as sns


def format_data(results_data, key="suppression"):
    formatted = {}
    for uuid, data in results_data.items():
        if formatted.get(data["relation_name"]) is None:
            formatted[data["relation_name"]] = {"related": [], "unrelated": []}

        related_data = data[key]["related"]
        related_change = []
        for prob in related_data["pct_change"]:
            related_change.append(prob)

        unrelated_data = data[key]["unrelated"]
        unrelated_change = []
        for prob in unrelated_data["pct_change"]:
            unrelated_change.append(prob)

        if data["n_refined_neurons"] > 0 and data["n_unrelated_neurons"] > 0:
            # for some prompts we didn't get any neurons back, it would be unfair to include them
            if related_change:
                related_change = sum(related_change) / len(related_change)
                if unrelated_change:
                    unrelated_change = sum(unrelated_change) / len(unrelated_change)
                else:
                    unrelated_change = 0.0
                formatted[data["relation_name"]]["related"].append(related_change)
                formatted[data["relation_name"]]["unrelated"].append(unrelated_change)

    for relation_name, data in formatted.items():
        if data["related"]:
            data["related"] = sum(data["related"]) / len(data["related"])
        else:
            data["related"] = float("nan")
        if data["unrelated"]:
            data["unrelated"] = sum(data["unrelated"]) / len(data["unrelated"])
        else:
            data["unrelated"] = float("nan")

    pandas_format = {"relation_name": [], "related": [], "pct_change": []}
    for relation_name, data in formatted.items():
        verb = "Suppressing" if key == "suppression" else "Enhancing"
        pandas_format["relation_name"].append(relation_name)
        pandas_format["pct_change"].append(data["related"])
        pandas_format["related"].append(f"{verb} knowledge neurons for related facts")

        pandas_format["relation_name"].append(relation_name)
        pandas_format["pct_change"].append(data["unrelated"])
        pandas_format["related"].append(f"{verb} knowledge neurons for unrelated facts")
    return pd.DataFrame(pandas_format).dropna()


def plot_data(pd_df, experiment_type, out_path="test.png"):
    sns.set_theme(style="whitegrid")
    if experiment_type == "suppression":
        title = "Suppressing knowledge neurons"
    elif experiment_type == "enhancement":
        title = "Enhancing knowledge neurons"
    else:
        raise ValueError
    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=pd_df,
        kind="bar",
        x="relation_name",
        y="pct_change",
        hue="related",
        ci="sd",
        palette="dark",
        alpha=0.6,
        height=6,
        aspect=4,
    )
    g.despine(left=True)
    g.set_axis_labels("relation name", "Correct probability percentage change")
    g.legend.set_title(title)
    g.savefig(out_path)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("Arguments for pararel result plotting")
    parser.add_argument(
        "--results_dir",
        default="bert_base_uncased_neurons/",
        type=str,
        help="directory in which the results from pararel_evaluate.py are saved.",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # load results
    result_paths = results_dir.glob("*results_*.json")
    results = {}
    for p in result_paths:
        with open(p) as f:
            results.update(json.load(f))

    # plot results of suppression experiment
    suppression_data = format_data(results, key="suppression")
    plot_data(suppression_data, "suppression", out_path="images/suppress.png")

    # plot results of enhancement experiment
    enhancement_data = format_data(results, key="enhancement")
    plot_data(enhancement_data, "enhancement", out_path="images/enhance.png")
