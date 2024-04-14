import yaml

from ..ChatBot import ChatBot
from ..constants import MODEL_CLIENT


class ClientBot(ChatBot):
    def __init__(self, version):
        self.system_prompt = self.load_prompt(version)
        self.conversation_history = [
            {"role": "system", "name": "client", "content": self.system_prompt}
        ]
        self.model = MODEL_CLIENT

    @property
    def session_id(self):
        return f"ClientBot-{id(self)}"

    def load_prompt(self, version):
        file_path = f"src/prompts/client/prompt_{version}.yaml"
        with open(file_path) as file:
            data = yaml.safe_load(file)
        prompt = data["text"]

        return prompt

    def converse(self, openai_obj, counsellor_turn=None):
        if counsellor_turn:
            self.conversation_history.append(counsellor_turn)

        response, usage = self.create_chat_message(
            openai_obj, self.conversation_history, self.model
        )
        self.conversation_history.append(
            {"role": "user", "name": "client", "content": response}
        )
        return response, usage
