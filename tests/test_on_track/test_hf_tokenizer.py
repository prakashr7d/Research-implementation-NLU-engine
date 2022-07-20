from pathlib import Path

from ruth.constants import TEXT
from ruth.nlu.tokenizer.hf_tokenizer import HFTokenizer
from ruth.shared.constants import ATTENTION_MASKS, INPUT_IDS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData


def test_hf_tokenizer(example_data_path: Path):
    training_data = TrainData.build(example_data_path)
    tokenizer = HFTokenizer.build({})
    tokenizer.train(training_data=training_data)
    assert training_data.training_examples[0].data[INPUT_IDS] == [101, 7632, 102, 0]
    assert training_data.training_examples[0].data[ATTENTION_MASKS] == [1, 1, 1, 0]
    message = RuthData(data={TEXT: "hello"})
    tokenizer.parse(message)
