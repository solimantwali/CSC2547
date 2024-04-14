import argparse
import csv
import json
import re
from pathlib import Path
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

patterns = [
    r"\(GI\)",
    r"\(Persuade\)",
    r"\(Persuade with\)",
    r"\(Q\)",
    r"\(SR\)",
    r"\(CR\)",
    r"\(AF\)",
    r"\(Seek\)",
    r"\(Emphasize\)",
    r"\(Confront\)",
]


def behavioural_counts(eval_file, counter_file, label_file):
    with open(eval_file) as file:
        data = json.load(file)

    aggr_metrics = {}
    aggr_metrics["name"] = "aggregated metrics"
    aggr_metrics["codes"] = {p: 0 for p in patterns}

    turn = 1
    for d in data[1:]:
        if d["name"] != "evaluator":
            data.remove(d)
            continue
        multi_encode = {p: 0 for p in patterns}
        for i, pattern in enumerate(patterns):
            if re.search(pattern, d["content"]):
                multi_encode[pattern] = 1
                if pattern == r"\(CR\)":
                    multi_encode[r"\(SR\)"] = 0
        aggr_metrics["codes"] = {
            x: aggr_metrics["codes"].get(x, 0) + multi_encode.get(x, 0)
            for x in patterns
        }
        d["codes"] = multi_encode
        d["turn"] = turn
        turn += 1

    aggr_codes = aggr_metrics["codes"]
    aggr_metrics["summary_scores"] = {}
    summ = aggr_metrics["summary_scores"]
    summ["R:Q"] = (aggr_codes[r"\(CR\)"] + aggr_codes[r"\(SR\)"]) / aggr_codes[r"\(Q\)"]
    summ["%CR"] = (
        (aggr_codes[r"\(CR\)"]) / (aggr_codes[r"\(CR\)"] + aggr_codes[r"\(SR\)"]) * 100
    )
    summ["Total MIA"] = (
        aggr_codes[r"\(AF\)"] + aggr_codes[r"\(Seek\)"] + aggr_codes[r"\(Emphasize\)"]
    )
    summ["Total MINA"] = (
        aggr_codes[r"\(Confront\)"]
        + aggr_codes[r"\(Persuade\)"]
        + aggr_codes[r"\(Persuade with\)"]
    )

    data[0] = aggr_metrics

    with open(counter_file, "w") as file:
        json.dump(data, file, indent=4)

    csv_header = ["Turn"] + [c.replace("\\", "") for c in list(data[0]["codes"].keys())]
    with open(label_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_header)  # Write the header

        for entry in data:
            if "codes" in entry and "turn" in entry:
                row = [entry["turn"]] + list(entry["codes"].values())
                csvwriter.writerow(row)


def compare_human_evalbot(bot_file, human_file, output_file):
    with open(bot_file) as file:
        ebot = json.load(file)
    with open(human_file) as file:
        human = json.load(file)

    comp = []
    for e, h in zip(ebot, human):
        # print(e)
        item = {}
        if e["name"] == "aggregated metrics" == h["name"]:
            item["aggregated metrics"] = {p: 0 for p in patterns}
        else:
            item["role"] = e["role"]
            item["name"] = e["name"]
            item["codes"] = {
                code: e["codes"][code] == h["codes"][code] for code in patterns
            }
            item["turn"] = e["turn"]
        comp.append(item)

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Turn", "Code", "EvalBot", "Human", "Match"])

        for e, h in zip(ebot, human):
            if e["name"] == "aggregated metrics" == h["name"]:
                continue
            for code in patterns:
                match = 0
                if h["codes"][code] == 1 and e["codes"][code] == 1:  # true positive
                    match = 1
                elif h["codes"][code] == 0 and e["codes"][code] == 0:  # true negative
                    match = 2
                elif h["codes"][code] == 0 and e["codes"][code] == 1:  # false positive
                    match = 3
                elif h["codes"][code] == 1 and e["codes"][code] == 0:  # false negative
                    match = 4
                writer.writerow(
                    [
                        e["turn"],
                        code.replace("\\", ""),
                        e["codes"][code],
                        h["codes"][code],
                        match,
                    ]
                )


def create_heatmap(comp, name, cols):
    df = pd.read_csv(comp)
    df = df.pivot_table(index="Turn", columns="Code", values="Match", sort=False)
    df.rename(
        columns={
            "(Persuade)": "(Per.)",
            "(Persuade with)": "(Per. with)",
            "(Emphasize)": "(Emph.)",
            "(Confront)": "(Conf.)",
        },
        inplace=True,
    )
    TP = df.apply(lambda x: (x == 1).sum())
    TN = df.apply(lambda x: (x == 2).sum())
    FP = df.apply(lambda x: (x == 3).sum())
    FN = df.apply(lambda x: (x == 4).sum())

    metrics = {
        "Accuracy": (TP + TN) / (TP + TN + FP + FN),
        "Precision": TP / (TP + FP),
        "Recall": TP / (TP + FN),
        "F1": TP / (TP + 0.5 * (FP + FN)),
    }

    if cols == 2:
        fig, axs = plt.subplots(
            3,
            2,
            figsize=(18, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        axs[0, 1].set_visible(False)
    elif cols == 1:
        fig, axs = plt.subplots(
            1 + len(metrics),
            1,
            figsize=(12, 3 + 3 * len(metrics)),
            sharex=True,
            gridspec_kw={"height_ratios": [4] + [1 for m in metrics]},
        )

    # First plot: Heatmap
    ax1 = axs[0, 0] if cols == 2 else axs[0]
    cmap = ListedColormap(["#4CAF50", "#2196F3", "#FFC107", "#F44336"])
    colors = [cmap(i) for i in range(cmap.N)]
    ax1.imshow(df, cmap=cmap, aspect="auto")
    labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
    patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=labels[i],
            markersize=10,
            markerfacecolor=colors[i],
        )
        for i in range(len(colors))
    ]
    ax1.set_yticks(np.arange(len(df.index)))
    ax1.set_xticks(np.arange(len(df.columns)))
    ax1.set_xticklabels(list(df.columns.values))
    ax1.xaxis.set_tick_params(which="both", labelbottom=True)
    ax1.set_yticklabels(df.index)
    ax1.set_title(f"{name}: Heatmap of Codes per Turn")
    ax1.set_ylabel("Turn")

    mlist = list(metrics.items())
    for i in range(len(mlist)):
        name, metric = mlist[i]
        ax = axs[1 + i // 2, i % 2] if cols == 2 else axs[1 + i]
        bar_positions = np.arange(len(df.columns) + 1)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(list(df.columns.values) + ["Average"])
        ax.set_title(name)
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=None, top=1.1)

        filnan = metric.fillna(0)
        bars = ax.bar(bar_positions[:-1], filnan, color="skyblue")
        avgbar = ax.bar(
            bar_positions[-1], metric.mean(), color="green", label="Average"
        )
        for i in range(len(bars)):
            bar = bars[i]
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval - 0.1 if bar.get_height() > 0 else 0,
                f"{yval*100:.2f}%" if not np.isnan(metric.iloc[i]) else "NaN",
                ha="center",
                va="bottom",
            )

        avgyval = avgbar[0].get_height()
        ax.text(
            avgbar[0].get_x() + avgbar[0].get_width() / 2,
            avgyval - 0.1 if avgbar[0].get_height() > 0 else 0,
            f"{avgyval*100:.2f}%",
            ha="center",
            va="bottom",
            color="w",
        )

    # Set common labels
    plt.xlabel("Codes")
    plt.tight_layout()
    ax1.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    ) if cols == 2 else ax1.legend(handles=patches, loc=1, borderaxespad=1.0)
    plt.savefig(str(comp)[:-4] + ".png", dpi=600)


def behavioural_counts_human(input_file, output_file):
    data = []
    aggr_metrics = {}
    aggr_metrics["name"] = "aggregated metrics"
    aggr_metrics["codes"] = {p: 0 for p in patterns}

    with open(input_file, encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            # Each row is already a dictionary
            item = {}
            item["role"] = "user"
            item["name"] = "evaluator"
            item["codes"] = {patterns[i]: int(row[1 + i]) for i in range(len(patterns))}
            item["turn"] = int(row[0])
            aggr_metrics["codes"] = {
                x: aggr_metrics["codes"].get(x, 0) + item["codes"].get(x, 0)
                for x in patterns
            }
            data.append(item)

    aggr_codes = aggr_metrics["codes"]
    aggr_metrics["summary_scores"] = {}
    summ = aggr_metrics["summary_scores"]
    summ["R:Q"] = (
        (aggr_codes[r"\(CR\)"] + aggr_codes[r"\(SR\)"]) / aggr_codes[r"\(Q\)"]
        if aggr_codes[r"\(Q\)"] > 0
        else "NaN"
    )
    summ["%CR"] = (
        (aggr_codes[r"\(CR\)"]) / (aggr_codes[r"\(CR\)"] + aggr_codes[r"\(SR\)"]) * 100
        if aggr_codes[r"\(CR\)"] > 0
        else "NaN"
    )
    summ["Total MIA"] = (
        aggr_codes[r"\(AF\)"] + aggr_codes[r"\(Seek\)"] + aggr_codes[r"\(Emphasize\)"]
    )
    summ["Total MINA"] = (
        aggr_codes[r"\(Confront\)"]
        + aggr_codes[r"\(Persuade\)"]
        + aggr_codes[r"\(Persuade with\)"]
    )

    data.insert(0, aggr_metrics)

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", required=True, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    folder = Path(args.folder)
    assert folder.exists() and folder.is_dir()

    assert (folder / "evaluator.json").exists()
    behavioural_counts(
        folder / "evaluator.json",
        folder / "counter.json",
        folder / "evalbot_labels.csv",
    )

    assert (folder / "human_labels_Anno2.csv").exists()
    behavioural_counts_human(
        folder / "human_labels_Anno2.csv", folder / "counter_Anno2.json"
    )
    compare_human_evalbot(
        folder / "counter.json",
        folder / "counter_Anno2.json",
        folder / "comparison_Anno2.csv",
    )
    create_heatmap(
        folder / "comparison_Anno2.csv",
        f"Full History Evaluator for {PurePath(folder)}: Annotator 2",
        2,
    )

    assert (folder / "human_labels_Soliman.csv").exists()
    behavioural_counts_human(
        folder / "human_labels_Soliman.csv", folder / "counter_soliman.json"
    )
    compare_human_evalbot(
        folder / "counter.json",
        folder / "counter_soliman.json",
        folder / "comparison_soliman.csv",
    )
    create_heatmap(
        folder / "comparison_soliman.csv",
        f"Full History Evaluator for {PurePath(folder)}: Annotator 1",
        2,
    )

    # behavioural_counts(
    #     folder / "evaluator_two_turns.json", folder / "counter_two_turns.json"
    # )
    # compare_human_evalbot(
    #     folder / "counter_two_turns.json",
    #     folder / "counter_human.json",
    #     folder / "comparison_two_turns.csv",
    # )
    # create_heatmap(folder / "comparison_two_turns.csv", "Last-two-turns evaluator", 1)
