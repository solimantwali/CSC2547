import argparse
from pathlib import Path
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


def IRR(folder, raters):
    n = len(raters)

    mtx = np.zeros((n, n))
    df = pd.DataFrame(mtx, columns=raters.keys(), index=raters.keys())
    keys = list(raters.keys())

    data = {}
    for i, val in enumerate(raters.values()):
        data[i] = pd.read_csv(val, index_col="Turn")

    columns = list(data[0].columns)
    columns.remove("(Persuade)")
    print(data)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))  # Adjust the figsize as needed
    axes_flat = axes.flatten()

    for k, column in enumerate(columns):
        for i in range(n):
            for j in range(n):
                # k1 = keys[i]
                # k2 = keys[j]
                # agree = f"100% agreement for {k1} and {k2} for {column}. K=1.0."
                # novar = f"No variance but <100% agreement for {k1} and {k2}. K=0.0."
                # try:
                #     ck = cohen_kappa_score(data[i][column], data[j][column])
                #     if np.isnan(ck):
                #         if np.all(data[i][column] == data[j][column]):
                #             ck = 1.0
                #             print(agree)
                #         else:
                #             ck = 0.0
                #             print(novar)
                # except ZeroDivisionError:
                #     ck = 0.0
                #     print(f"ZeroDivisionError for {keys[i]} & {keys[j]}. Kappa=0.0.")
                ck = cohen_kappa_score(data[i][column], data[j][column])
                mtx[i][j] = ck
                print(f"kappa between {keys[i]} and {keys[j]} for {column}:", ck)

        mask = np.triu(np.ones_like(df, dtype=bool), k=0)
        sns.heatmap(
            df,
            annot=True,
            ax=axes_flat[k],
            mask=mask,
            fmt=".2f",
            vmax=1.0,
            vmin=-1.0,
            cbar=True,
        )
        axes_flat[k].set_title(f"{column}")
        axes_flat[k].set_xticklabels(axes_flat[k].get_xticklabels(), rotation=0)

    plt.suptitle(
        r"Cohen's $\kappa$ per code between raters for " + f"{PurePath(folder)}"
    )
    plt.tight_layout()
    plt.savefig(folder / "cohens_kappa_nan.png", dpi=600)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", required=True, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    folder = Path(args.folder)
    assert folder.exists() and folder.is_dir()

    raters = {
        "Anno. 1": folder / "human_labels_Soliman.csv",
        "Anno. 2": folder / "human_labels_Anno2.csv",
        "EvalBot": folder / "evalbot_labels.csv",
    }
    assert [r.exists() for r in raters.values()]

    IRR(folder, raters)
