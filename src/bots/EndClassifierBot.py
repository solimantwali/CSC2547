import yaml

from ..ChatBot import ChatBot
from ..constants import MODEL_END_CLS


class EndClassifierBot(ChatBot):
    def __init__(self, version):
        self.system_prompt = self.load_prompt(version)
        self.system_prompt_dict = [
            {"role": "system", "name": "end_classifier", "content": self.system_prompt}
        ]
        self.model = MODEL_END_CLS

    @property
    def session_id(self):
        return f"EndClassifierBot-{id(self)}"

    def load_prompt(self, version):
        file_path = f"src/prompts/end_classifier/prompt_{version}.yaml"
        with open(file_path) as file:
            data = yaml.safe_load(file)
        prompt = data["text"]

        return prompt

    def converse(self, openai_obj, input):
        message_dict = self.system_prompt_dict
        message_dict.append(input)

        response, usage = self.create_chat_message(openai_obj, message_dict, self.model)

        return response, usage
