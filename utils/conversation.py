import dataclasses
from enum import auto, Enum
from typing import List, Any


class SeparatorStyle(Enum):
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    PHOENIX = auto()
    MINICHAT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separator
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    # TODO(lmzheng): refactor this
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.BAIZE:
            ret = self.system + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + message + "\n"
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.MINICHAT:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + " " + message + self.sep
                else:
                    ret += role # No space is needed.
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }


conv_minichat = Conversation(
    system='You are the parts of the human brain that conduct a dialogue, you can ' \
        'enter into verbal altercations with the interlocutor. You need to response emotionally.',
    roles=("[|User|]", "[|Assistant|]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MINICHAT,
    sep="</s>",
)

conv_phi = Conversation(
    system='You are the parts of the human brain that conduct a dialogue, you can ' \
        'enter into verbal altercations with the interlocutor. You need to response emotionally.\n',
    roles=("[|User|]", "[|Assistant|]"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MINICHAT,
    sep="<|endoftext|>\n",
)

conv_templates = {
    "minichat": conv_minichat,
    "phi": conv_phi,
}

def get_default_conv_template(model_name):
    model_name = model_name.lower()
    try:
        ret = conv_templates[model_name]
        return ret.copy()
    except:
        raise NotImplementedError(f"No support for model {model_name}.")
    

if __name__ == "__main__":
    conv = conv_templates["minichat"].copy()
    conv.append_message(conv.roles[0], "Write a Python function that checks if a given number is even or odd.")
    conv.append_message(conv.roles[1], None)
    print([conv.get_prompt()])
