import re

import yaml
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
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
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex



from ..ChatBot import ChatBot
from ..constants import MODEL_EVALUATOR
from ..constants import PERSIST_DIR


class EvalBot(ChatBot):
    def __init__(self, version, few_shot=False):
        self.version = version
        self.system_prompt = self.load_prompt(version) #########################################################################################
        self.conversation_history = [
            {"role": "system", "name": "evaluator", "content": self.system_prompt}
        ]
        self.prompt_only = [
            {"role": "system", "name": "evaluator", "content": self.system_prompt}
        ]
        self.model = MODEL_EVALUATOR
        self.query_engine = self.assemble_engine(k=10) if few_shot else None
        self.to_csv = []

    @property
    def session_id(self):
        return f"EvalBot-{id(self)}"

    def assemble_engine(self, k):
        Settings.llm = OpenAI(model="gpt-4-0125-preview")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context, show_progress=True)
        # filters = MetadataFilters(
        #     filters=[
        #         ExactMatchFilter(key="Utterance", value="TRUE"),
        #         ExactMatchFilter(key="Context", value="FALSE"),
        #     ]
        # )

        # keyword_index = SimpleKeywordTableIndex(storage_context=storage_context)

        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=k,
        )
        # keyword_retriever = KeywordTableSimpleRetriever(index=vector_index)

        # retriever=CustomRetriever(vector_retriever, keyword_retriever)

        # assemble query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=vector_retriever,
            response_mode="no_text",
            node_postprocessors=[SimilarityPostprocessor()],
        )

        return query_engine

    def load_prompt(self, version):
        file_path = f"src/prompts/evaluator/prompt_{version}.yaml"
        with open(file_path) as file:
            data = yaml.safe_load(file)
        prompt = data["text"]

        return prompt

    def converse(self, openai_obj, counsellor_turn=None, client_turn=None):
        if client_turn:
            self.conversation_history.append(client_turn)
        if counsellor_turn:
            self.conversation_history.append(counsellor_turn)

        response, usage = self.create_chat_message(
            openai_obj, self.conversation_history, self.model
        )
        self.conversation_history.append(
            {"role": "user", "name": "evaluator", "content": response}
        )
        return response, usage

    def evaluate(self, openai_obj, last_two_turns, greedy=False):
        history = self.prompt_only.copy()

        # print('ltt:', last_two_turns)
        if self.query_engine:
            # print('fewshot mode onzo')
            response = self.query_engine.query(last_two_turns[-1]["content"])
            # print(last_two_turns[-1]["content"])
            # print(response.source_nodes[0].text)
            for node in response.source_nodes:
                history.extend(
                    [
                        {"role": "user", "name": "counsellor", "content": node.text},
                        {
                            "role": "system",
                            "name": "evaluator",
                            "content": node.metadata["Label"],
                        },
                    ]
                )
            # print('TOP MATCH:\n', history[1])


        history.extend(last_two_turns)
        # print('MESSAGES:\n' + '\n'.join(str(i) for i in history[1:]))
        print('messages: ', [i for i in history])

        self.conversation_history.extend(last_two_turns)
        # print(self.conversation_history)

        # exit()
        if greedy:
            response, usage = self.create_chat_message_greedy(
                openai_obj, history, self.model
            )
        else:
            response, usage = self.create_chat_message(openai_obj, history, self.model)

        print("response: ", response)

        self.conversation_history.append(
            {"role": "user", "name": "evaluator", "content": response}
        )

        if len(last_two_turns) > 1:
            predicted_bc = ""
            try:
                predicted_bc = re.findall(r"\(([^)]+)\)", response)[0]
            except IndexError:
                predicted_bc = response.strip()

            self.to_csv.append([last_two_turns[-1]["content"], predicted_bc])

        return response, usage
