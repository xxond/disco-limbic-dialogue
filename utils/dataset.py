import copy
import torch
from torch.utils.data import Dataset
from utils.conversation import get_default_conv_template

MODEL_TYPES = [
    'minichat',
    'phi'
]
INST = 'You are the parts of the human brain that ' \
        'conduct a dialogue, you can enter into verbal ' \
        'altercations with the interlocutor. You need to response emotionally.'


# Minichat
def template(dial, model_type):
    conv = get_default_conv_template(model_type)
    for i in dial:
        if i[0] == '[':
            conv.append_message(conv.roles[1], i)
        else:
            conv.append_message(conv.roles[0], i)
    return conv.get_prompt()


class CombineDataset(Dataset):
    def __init__(self, data, tokenizer,
                 partition="train",
                 max_words=512,
                 model_type='minichat'):
        assert model_type in MODEL_TYPES, f'model type {model_type} not supported'
        self.model_type = model_type
        self.ann = data

        self.ann = self.ann
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100

        ann = self.ann[index]
        prompt = template(ann[:-1], self.model_type)
        example = template(ann, self.model_type)

        prompt = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False),
                              dtype=torch.int64,)
        example = self.tokenizer.encode(example, add_special_tokens=False)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example_len = len(example)
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) + self.tokenizer.pad_token_id))
        elif padding <= 0:
            example = example[: self.max_words]
            example_len = len(example)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        labels[example_len: ] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        #example[~example_mask] = 0
        labels[~label_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask[example_len: ] = False
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
