import json
from pathlib import Path
from typing import Any, Dict, List, Text

import pytest
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def example_data(example_data_path: Path) -> List[Dict[Text, Any]]:
    with open(example_data_path, "r") as file_pointer:
        return json.load(file_pointer)


def test_collections_positive(example_data: List[Dict[Text, Any]]):
    messages = []

    for data in example_data:
        messages.append(RuthData(data=data))

    training_data = TrainData(messages)

    assert len(messages) == len(training_data)
