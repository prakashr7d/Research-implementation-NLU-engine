from pathlib import Path
from typing import Any, Dict, Text

import pytest
from ruth.constants import PATH, TEXT, TOKENS
from ruth.nlu.tokenizer.tokenizer import Token

FEATURE = "feature"
TOKEN = "token"


@pytest.fixture
def example_data_path() -> Path:
    return Path("data/test/ruth_example_data/training_example.json")


@pytest.fixture
def count_featurizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "I am a developer",
        FEATURE: [[0, 0, 0]],
        PATH: Path("data/test/ruth_example_data/training_example.json"),
    }


@pytest.fixture
def whitespace_example() -> Dict[Text, Any]:
    return {
        TEXT: "I am a developer",
        TOKENS: [
            Token("I", start=0, end=1),
            Token("am", start=2, end=4),
            Token("a", start=5, end=6),
            Token("developer", start=7, end=16),
        ],
        PATH: Path("data/test/ruth_example_data/training_example.json"),
    }


@pytest.fixture
def example_classifier_data() -> Path:
    return Path("data/test/classification/classification_data.json")


@pytest.fixture
def bert_tokenizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "He lived HapPily",
        TOKEN: [[0, 0, 0]],
        PATH: Path("data/test/ruth_example_data/training_example.json"),
    }


@pytest.fixture
def tfidf_featurizer_example() -> Dict[Text, Any]:
    return {
        TEXT: "I am a developer",
        FEATURE: [[0, 0, 0]],
        PATH: Path("data/test/ruth_example_data/training_example.json"),
    }
