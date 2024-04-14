import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def main(input_file, output_folder):
    # Extract the content after 'result_' and before '.csv' from the input filename
    match = re.search(r"result_(.*?)\.csv", input_file)
    if match:
        content = match.group(1)
        print(f"Content extracted from filename: {content}")
    elif input_file == "result.csv":
        content = ""
    else:
        print("No content found or incorrect file name format.")
        sys.exit(1)

    # Load the CSV file
    # output_folder = Path("data/simple-conversation-038/")
    df = pd.read_csv(output_folder / input_file)

    # Extract the 'Output' and 'Label' columns
    y_pred = df["Output"]
    y_true = df["Label"]

    # Compute the classification report and confusion matrix
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Determine the unique classes for labeling purposes
    classes = df["Label"].unique()
    classes.sort()  # Sort the class labels

    # Open a text file to save the output
    if not content:
        filename = output_folder / "classifier_performance_report.txt"
    else:
        filename = output_folder / f"classifier_performance_report_{content}.txt"
    with open(filename, "w") as f:
        # Write the classification report and confusion matrix to the file
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix {content.capitalize()}")

    # Adjust layout to make room for the labels
    plt.tight_layout()

    # Save the confusion matrix plot to a file
    if not content:
        filename = output_folder / "confusion_matrix.png"
    else:
        filename = output_folder / f"confusion_matrix_{content}.png"
    plt.savefig(filename)
    plt.close()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True, type=str)

args = parser.parse_args()

if __name__ == "__main__":
    file = Path(args.file)
    assert file.exists()

    main(file.name, file.parent)
