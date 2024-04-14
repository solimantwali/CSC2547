from abc import ABC
from abc import abstractproperty


class ChatBot(ABC):
    @abstractproperty
    def session_id(self):
        pass

    def create_chat_message(self, openai_obj, messages, model):
        response = openai_obj.chat.completions.create(
            model=model, messages=messages, user=self.session_id
        )
        return response.choices[0].message.content, response.usage

    def create_chat_message_greedy(self, openai_obj, messages, model):
        response = openai_obj.chat.completions.create(
            model=model, messages=messages, user=self.session_id, temperature=0
        )
        return response.choices[0].message.content, response.usage
