import json
from pathlib import Path

import pytest
from ruth.constants import INTENT, TEXT
from ruth.nlu.classifiers.svm_classifier import SVMClassifier
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


@pytest.fixture
def classifier_data(example_classifier_data: Path) -> TrainData:
    with open(example_classifier_data, "r") as f:
        examples = json.load(f)

    training_data = TrainData()
    for value in examples:
        training_data.add_example(RuthData(value))

    return training_data


def test_svm_classifier(
    classifier_data: TrainData,
):
    ftr = CountVectorFeaturizer({})
    ftr.train(classifier_data)

    classifier = SVMClassifier({})
    classifier.train(training_data=classifier_data)
    message = RuthData({TEXT: "hello"})
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "ham"

    message = RuthData(
        {
            TEXT: "WINNER!! As a valued network customer you have been"
            " selected to received £900 prize reward! "
            "To claim call 09061701461. Claim code KL341."
            " Valid 12 hours only."
        }
    )
    ftr.parse(message)
    classifier.parse(message)
    assert message.get(INTENT)["name"] == "spam"
