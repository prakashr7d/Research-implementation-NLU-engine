import json
from typing import Any, Dict, Text

from ruth.constants import PATH, TEXT, TOKENS
from ruth.nlu.tokenizer.whitespace_tokenizer import WhiteSpaceTokenizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_with_ham_spam_with_empty_training_data(whitespace_example: Dict[Text, Any]):
    messages = []
    with open(whitespace_example[PATH], "r") as f:
        example_data = json.load(f)
    for data in example_data:
        messages.append(RuthData(data=data))
    training_data = TrainData(messages)
    tokenizer = WhiteSpaceTokenizer({})
    tokenizer.train(training_data)
    test_message = RuthData.build(text=whitespace_example[TEXT])
    tokenizer.parse(test_message)

    assert whitespace_example[TOKENS] == test_message.get(TOKENS)
