import json
from pathlib import Path
from typing import List, Text

from ruth.constants import INTENT, TEXT
from ruth.shared.nlu.training_data.ruth_data import RuthData


class TrainData:
    def __init__(self, training_examples: List[RuthData] = None):
        self.training_examples = training_examples or []

    def __len__(self) -> int:
        return len(self.training_examples)

    @classmethod
    def build(cls, data_path: Path) -> "TrainData":
        training_examples = list()
        with open(data_path, "r") as f:
            messages = json.load(f)
        for message in messages:
            training_examples.append(
                RuthData.build(intent=message.get(INTENT), text=message.get(TEXT))
            )
        return cls(training_examples)

    def add_example(self, data: RuthData) -> List[RuthData]:
        self.training_examples.append(data)
        return self.training_examples

    def __getitem__(self, item):
        return self.training_examples[item]

    @staticmethod
    def get_text_list(training_examples: List[RuthData]) -> List[Text]:
        return [example.get(TEXT) for example in training_examples]

    @property
    def intent_examples(self) -> List[RuthData]:
        """Returns the list of examples that have intent."""
        return [ex for ex in self.training_examples if ex.get(INTENT)]
