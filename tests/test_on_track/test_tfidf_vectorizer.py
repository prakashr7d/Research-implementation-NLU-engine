import json
from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT
from ruth.nlu.featurizers.tfidf_featurizers.tfidf_vector_featurizer import (
    TfidfVectorFeaturizer,
)
from ruth.shared.nlu.ruth_elements import RuthData, TrainData

from tests.test_on_track.conftest import FEATURE


def test_tfidf_vectorizer(tfidf_featurizer_example: Dict[Text, Any]):
    messages = []
    with open(tfidf_featurizer_example[PATH], "r") as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data=data))
    training_data = TrainData(messages)

    featurizer = TfidfVectorFeaturizer({})
    featurizer.train(training_data)
    test_message = RuthData.build(text=tfidf_featurizer_example[TEXT])
    featurizer.parse(test_message)

    assert (
        tfidf_featurizer_example[FEATURE]
        == test_message.get_sparse_features().feature.toarray().tolist()
    )
