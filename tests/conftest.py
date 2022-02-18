from pathlib import Path
from typing import Any, Dict, Text

import pytest
from ruth.constants import PATH, TEXT

FEATURE = "feature"


@pytest.fixture
def example_data_path() -> Path:
    return Path("data/test/ruth_example_data/training_example.json")


@pytest.fixture
def count_featurizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "I am a developer",
        FEATURE: [[0, 0]],
        PATH: Path("data/test/ruth_example_data/training_example.json"),
    }


@pytest.fixture
def example_classifier_data() -> Path:
    return Path("data/test/ruth_example_data/training_example.json")
