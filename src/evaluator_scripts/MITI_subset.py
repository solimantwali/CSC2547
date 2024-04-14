import numpy as np
import pandas as pd


if __name__ == "__main__":
    n = 40
    # Set the seed for reproducibility
    np.random.seed(42)

    # Load the source CSV file
    source_df = pd.read_csv(
        "data/MITI4_2/miti4_2_BC_data.csv", encoding="MacRoman", dtype="str"
    )
    print(source_df)

    # Define the condition: "Utterance" is TRUE and "Context" is FALSE
    condition = (source_df["Utterance"] == "TRUE") & (source_df["Context"] == "FALSE")

    # Filter the DataFrame based on the condition
    filtered_df = source_df[condition]
    print(filtered_df)

    if len(filtered_df) > n:
        filtered_df = filtered_df.sample(n=n)

    # Save the selected rows to the target CSV file
    filtered_df.to_csv(f"data/MITI4_2/MITI_subset_{n}.csv", index=False)
