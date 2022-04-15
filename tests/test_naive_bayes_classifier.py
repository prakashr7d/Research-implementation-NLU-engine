import json
from pathlib import Path

import pytest
from ruth.constants import INTENT, TEXT
from ruth.nlu import NaiveBayesClassifier
from ruth.nlu.featurizers import CountVectorFeaturizer
from ruth.shared import RuthData, TrainData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:
    with open(example_classifier_data, "r") as f:
        examples = json.load(f)

    training_data = TrainData()
    for value in examples:
        training_data.add_example(RuthData(value))

    return training_data


def test_naive_bayes_classifier(
    classifier_data: TrainData,
):
    ftr = CountVectorFeaturizer({})
    ftr.train(classifier_data)

    classifier = NaiveBayesClassifier({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello"})
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "ham"
    assert message.get(INTENT)["accuracy"] == 1.0
