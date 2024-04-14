import random

import yaml

from ..ChatBot import ChatBot
from ..constants import MODEL_COUNSELLOR


class CounsellorBot(ChatBot):
    def __init__(self, version, client_name, sample=False):
        self.sample = sample
        self.system_prompt = self.load_prompt(version, client_name)
        self.conversation_history = [
            {"role": "system", "name": "counsellor", "content": self.system_prompt}
        ]
        # TODO: Figure out how to implement Persuade with properly
        self.bc_list = [
            "Giving Information (GI)",
            # "Persuade with Permission (Persuade with)",
            "Question (Q)",
            "Simple Reflection (SR)",
            "Complex Reflection (CR)",
            "Affirm (AF)",
            "Seeking Collaboration (Seek)",
            "Emphasizing Autonomy (Emphasize)",
            "Not Coded (NC)",
        ]
        self.bc_list_abv = ["GI", "Q", "SR", "CR", "AF", "Seek", "Emphasize", "NC"]
        self.bc_prompts = self.load_bc_prompts()
        self.bc_history = []
        self.to_csv = []
        self.model = MODEL_COUNSELLOR
        self.end = False

    @property
    def session_id(self):
        return f"CounsellorBot-{id(self)}"

    def load_prompt(self, version, client_name):
        file_path = f"src/prompts/counsellor/prompt_{version}.yaml"
        with open(file_path) as file:
            data = yaml.safe_load(file)
        prompt = data["text"]
        prompt = prompt.format(client_name=client_name)
        return prompt

    def load_bc_prompts(self):
        bc_prompts = dict()
        for bc_abv in self.bc_list_abv:
            file_path = f"src/prompts/counsellor/prompt_{bc_abv}.yaml"
            with open(file_path) as file:
                data = yaml.safe_load(file)
            bc_prompts[bc_abv] = data["text"]
        return bc_prompts

    def is_end(self, turns):
        # Roles a die to see if we're ending the conversation
        if turns < 10:
            return False
        elif turns >= 50:
            return True
        else:
            # Calculate the probability linearly
            probability = (turns - 10) / (50 - 10)
            return random.random() < probability

    def converse(self, openai_obj, client_turn=None):
        if client_turn:
            self.conversation_history.append(client_turn)

            if self.sample:
                # Do sampling only if client has already said things
                if not self.end:
                    print('bc history:', self.bc_history)
                    index = random.randint(0, 7)
                    bc = self.bc_list[index]
                    bc_abv = self.bc_list_abv[index]
                    self.bc_history.append(bc_abv)

                    if index != 7:
                        # system_msg = (
                        #     "For your next turn of speech, "
                        #     "provide a single utterance of the {} category.\n\n"
                        # )
                        # system_msg = system_msg.format(bc)
                        # system_msg = system_msg + self.bc_prompts[bc_abv]
                        system_msg = (
                            "For your next turn of speech, provide an appropriate single "
                            "utterance as per one of the following MITI 4.2.1 behavioural codes (not the code itself):\n"
                            "1. Giving Information (GI)\n"
                            "2. Persuade (Persuade) \n"
                            "3. Persuade with Permission (Persuade with)\n"
                            "4. Question (Q)\n"
                            "5. Simple Reflection (SR)\n"
                            "6. Complex Reflection (CR)\n"
                            "7. Affirm (AF)\n"
                            "8. Seeking Collaboration (Seek)\n"
                            "9. Emphasizing Autonomy (Emphasize)\n"
                            "10. Confront (Confront)\n"
                        )
                        # print(system_msg)
                    else:
                        system_msg = self.bc_prompts[bc_abv]

                    # Roles a die to see if we're ending the conversation next turn
                    self.end = self.is_end(len(self.bc_history))
                else:
                    bc_abv = "NC"
                    self.bc_history.append(bc_abv)
                    system_msg = (
                        "For your next turn of speech, "
                        "naturally end the conversation in a polite manner."
                    )

                self.conversation_history.append(
                    {"role": "system", "name": "counsellor", "content": system_msg}
                )
                response, usage = self.create_chat_message(
                    openai_obj, self.conversation_history, self.model
                )
                # Remove system prompt from conversation history
                _ = self.conversation_history.pop()
                self.conversation_history.append(
                    {"role": "user", "name": "counsellor", "content": response}
                )
                self.to_csv.append([response, bc_abv])

                return response, usage

        response, usage = self.create_chat_message(
            openai_obj, self.conversation_history, self.model
        )
        self.conversation_history.append(
            {"role": "user", "name": "counsellor", "content": response}
        )
        return response, usage
