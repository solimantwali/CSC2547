import argparse

import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--transcript_id", required=True, type=int)


args = parser.parse_args()

if __name__ == "__main__":
    convo = args.transcript_id

    df = pd.read_csv(
        "data/AnnoMI-full.csv",
        usecols=["transcript_id", "interlocutor", "utterance_text"],
    )
    jfn = "data/Anno_MI-" + str(convo) + "/counsellor.json"

    filt = df[df["transcript_id"] == 131]
    filt = filt.drop("transcript_id", axis=1)
    filt.rename(
        columns={"interlocutor": "name", "utterance_text": "content"}, inplace=True
    )
    filt["name"] = filt["name"].str.replace("therapist", "counsellor", case=False)

    filt.insert(0, "role", "user")

    jstr = filt.to_json(orient="records", indent=4)

    with open(jfn, "w") as json_file:
        json_file.write(jstr)
