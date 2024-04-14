import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from llama_index.core import Document
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores import ExactMatchFilter
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List

# PERSIST_DIR = "data/csv_transcript_persist"
PERSIST_DIR = "data/csv_transcript_persist_miti_textnode"
PERSIST_DIR_KW = "data/csv_transcript_persist_miti_textnode_kw"

Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

def create_index(folder):
    doclist = []
    for file in folder.iterdir():
        print(file.name)
        if file.name == "miti4_2_BC_data.csv":
            with open(file, encoding="MacRoman") as csvfile:
                reader = csv.DictReader(csvfile)
                next(reader)
                for row in reader:
                    # print(row)
                    if row["Utterance"] == "TRUE" and row["Context"] == "FALSE":
                        document = TextNode(
                            text=row["Sentence"],
                            metadata={
                                key: value for key, value in list(row.items())[1:-1]
                            },
                            excluded_embed_metadata_keys=[
                                "Utterance",
                                "Volley",
                                "Context",
                            ],
                            excluded_llm_metadata_keys=[
                                "Utterance",
                                "Volley",
                                "Context",
                            ],
                            text_template="({metadata_str}) {content}",
                        )
                        print(document.get_content(metadata_mode=MetadataMode.EMBED))
                        doclist.append(document)
        elif file.name == "MI_Dataset_filtered_asdf.csv":
            df = pd.read_csv(file)
            doclist_new = df.apply(
                lambda row: Document(
                    text=row["text"],
                    metadata={"Label": row["final agreed label"]},
                    text_template="({metadata_str}) {content}",
                ),
                axis=1,
            ).tolist()
            for document in doclist_new:
                print(document.get_content(metadata_mode=MetadataMode.EMBED))

            doclist.extend(doclist_new)
        elif file.name == "MI_Dataset_asdf.csv":
            repl_dict = {
                "Affirm": "AF",
                "Closed Question": "Q",
                "Open Question": "Q",
                "Simple Reflection": "SR",
                "Complex Reflection": "CR",
                "Give Information": "GI",
                "Advise with Permission": "Persuade with",
                "Advise without Permission": "Persuade",
                "Emphasize Autonomy": "Emphasize",
            }
            df = pd.read_csv(file)
            df = df[df["author"] != "speaker"]
            df = df[
                ~df["final agreed label"].isin(
                    ["-", "N/A", "Support", "Direct", "Self-Disclose", "Warn", "Other"]
                )
            ]
            df["final agreed label"] = df["final agreed label"].replace(repl_dict)
            df.drop(
                columns=["dialog_id", "title", "turn_count", "turn", "author"],
                inplace=True,
            )
            # df.to_csv(folder/"MI_Dataset_filtered.csv", index=False)
            doclist_new = df.apply(
                lambda row: Document(
                    text=row["text"],
                    metadata={"Label": row["final agreed label"]},
                    text_template="({metadata_str}) {content}",
                ),
                axis=1,
            ).tolist()
            doclist.extend(doclist_new)
    return doclist


def fix_label(label):
    repl_dict = {
        "Persuade": "Per.",
        "Persuade with": "Per. with",
        "Emphasize": "Emph.",
        "Confront": "Conf.",
    }
    # Replace each phrase with its corresponding value
    for key in repl_dict:
        label = label.replace(key, repl_dict[key])

    return label

def get_luminance(color):
    # Assuming color is RGB
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

# Function to choose text color based on luminance
def get_text_color(bkg_color):
    luminance = get_luminance(bkg_color)
    return "white" if luminance < 0.65 else "black"

def heatmap(results):
    data = pd.read_csv(results, encoding="MacRoman")
    pivot_table = data.pivot_table(
        index=["Input Label", "Node Rank"], values="Node Similarity", aggfunc="first"
    ).unstack()
    # print(data.axes)
    # repl_dict={
    #     "Persuade": "Per.",
    #     "Persuade with": "Per. with",
    #     "Emphasize": "Emph.",
    #     "Confront": "Conf.",
    # }
    # data["Node Label"] = data["Node Label"].replace(repl_dict)

    # Create the heatmap
    # plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    heatmap = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": "Similarity Score"},
        ax=ax,
    )
    norm = Normalize(
        vmin=pivot_table.min().min(), vmax=pivot_table.max().max()
    )  # Normalize the scale based on your data
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Annotating each cell with the corresponding Node Label
    for i, row in enumerate(pivot_table.iterrows()):
        for j, val in enumerate(row[1]):
            # print(val)
            node_label = data.loc[
                (data["Input Label"] == row[0]) & (data["Node Rank"] == j + 1),
                "Node Label",
            ]
            if not node_label.empty:
                text = fix_label(node_label.values[0])
                heatmap.text(
                    j + 0.5,
                    i + 0.8,
                    text,
                    ha="center",
                    va="bottom",
                    color=get_text_color(sm.to_rgba(val)),
                )

    pivot_labels = data.pivot_table(
        index=["Input Label", "Node Rank"], values="Node Label", aggfunc="first"
    ).unstack()
    for i, (input_label, row) in enumerate(pivot_labels.iterrows()):
        for j, node_label in enumerate(row):
            # print(node_label, input_label[0])
            if pd.notnull(node_label) and node_label == input_label:
                rect = Rectangle(
                    (j + 0.05, i + 0.05), 0.9, 0.9, fill=False, color="yellow", lw=3
                )
                ax.add_patch(rect)
    node_ranks = [rank for rank in pivot_table.columns.get_level_values(1)]
    ax.set_xticks([x + 0.5 for x in range(len(node_ranks))])  # +0.5 centers the ticks
    ax.set_xticklabels(node_ranks)
    plt.title("Heatmap of Node Similarity by Input Label and Node Rank")
    plt.xlabel("Node Rank")
    plt.ylabel("Input Label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder", required=True, type=str)
args = parser.parse_args()

if __name__ == "__main__":
    folder = Path(args.folder)
    assert folder.exists()

    # build index
    ppath_exists = Path(PERSIST_DIR).exists()
    ppath_isempty = Path(PERSIST_DIR).is_dir() and not any(Path(PERSIST_DIR).iterdir())
    print(
        f'{PERSIST_DIR} does {"not " if not ppath_exists else ""}exist '
        f'and is {"not " if not ppath_isempty else ""}empty'
    )

    if not ppath_exists or ppath_isempty:
        print("storing to disk...\n")
        doclist = create_index(folder)
        vector_index = VectorStoreIndex(doclist, show_progress=True, llm=Settings.llm)
        vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
        kw_index = SimpleKeywordTableIndex(doclist, show_progress=True, llm=Settings.llm)
        kw_index.storage_context.persist(persist_dir=PERSIST_DIR_KW)
    else:
        print("loading from disk...\n")
        storage_context_v = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context_v, show_progress=True)
        storage_context_kw = StorageContext.from_defaults(persist_dir=PERSIST_DIR_KW)
        kw_index = load_index_from_storage(storage_context_kw, show_progress=True)



    # configure retriever
    k = 20
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="Utterance", value="TRUE"),
            ExactMatchFilter(key="Context", value="FALSE"),
        ]
    )
    # define custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=k)
    keyword_retriever = KeywordTableSimpleRetriever(index=kw_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

    # define response synthesizer
    # response_synthesizer = get_response_synthesizer()

    # assemble query engine
    # custom_query_engine = RetrieverQueryEngine(
    #     retriever=custom_retriever,
    #     response_synthesizer=response_synthesizer,
    # )

    # vector query engine
    # vector_query_engine = RetrieverQueryEngine(
    #     retriever=vector_retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    # # keyword query engine
    # keyword_query_engine = RetrieverQueryEngine(
    #     retriever=keyword_retriever,
    #     response_synthesizer=response_synthesizer,
    # )

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=custom_retriever,
        response_mode="no_text",
        # vector_store_query_mode="mmr",
        # node_postprocessors=[SimilarityPostprocessor()],
        # node_postprocessors=[LLMRerank(top_n=10, llm=Settings.llm)],

    )

    with open(folder / "test_set.csv", encoding="MacRoman") as rfile, open(
        folder / "results.csv", "w", encoding="MacRoman"
    ) as wfile:
        reader = csv.reader(rfile)
        writer = csv.writer(wfile)
        next(reader)
        writer.writerow(
            [
                "Input Text",
                "Input Label",
                "Node Rank",
                "Node Similarity",
                "Node Label",
                "Node Text",
            ]
        )
        for row in reader:
            query = row[0]
            print(f"query: {query}")

            response = query_engine.query(query)

            input_label_formatted = f"\tInput Label: \033[93m{row[1]:<13}\033[0m"
            output_label_intro = f" => Output Labels (k={k}): "

            # Prepare the list of labels with conditional formatting
            labels_list_str = ", ".join(
                [
                    f"\033[93m{node.metadata['Label']}\033[0m"
                    if row[1] in node.metadata["Label"]
                    else node.metadata["Label"]
                    for node in response.source_nodes
                ]
            )

            final_message = (
                f"{input_label_formatted}{output_label_intro}{labels_list_str}"
            )

            print(final_message)

            # print(f"Top k={k} similar nodes:")
            for i in range(len(response.source_nodes)):
                node = response.source_nodes[i]
                print([f for f in node])
                # if i == 0:
                writer.writerow(
                    [
                        query,
                        row[1],
                        i + 1,
                        f"{node.score:0.4f}" if node.score else "-1.0",
                        node.metadata["Label"],
                        node.text,
                    ]
                )
                # else:
                #     writer.writerow(
                #         [
                #             None,
                #             None,
                #             i + 1,
                #             f"{node.score:0.4f}",
                #             node.metadata["Label"],
                #             node.text,
                #         ]
                #     )

    heatmap(folder / "results.csv")
    print("hello llama")
