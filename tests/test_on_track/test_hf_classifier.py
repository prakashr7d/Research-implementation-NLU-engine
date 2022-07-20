from pathlib import Path

from ruth.constants import TEXT
from ruth.nlu.classifiers.hf_classifier import HFClassifier
from ruth.nlu.tokenizer.hf_tokenizer import HFTokenizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_hf_classifier(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = HFTokenizer.build({})
    tokenizer.train(training_data=training_data)
    classifier = HFClassifier.build({})
    classifier.train(training_data=training_data)
    # classifier.persist("trial", "D:/others/ruth/Research-implementation-NLU-engine/saved_models")
    # tokenizer.persist("trial", "D:/others/ruth/Research-implementation-NLU-engine/saved_models")
    message = RuthData(data={TEXT: "hello"})
    tokenizer.parse(message)
    classifier.parse(message)
    print("hi")
