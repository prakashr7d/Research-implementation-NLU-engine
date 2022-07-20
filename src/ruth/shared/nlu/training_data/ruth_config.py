import copy
from typing import Any, Dict, List, Text

from ruth.nlu.constants import INDEX
from ruth.shared.constants import ELEMENT_INDEX, KEY_LANGUAGE


@property
def default_pipline() -> List[Dict[Text, Any]]:
    return [{"name": "CountVectorFeaturizer"}, {"name": "NaiveBayesClassifier"}]


class RuthConfig:
    def __init__(self, config: Dict[Text, Any]):
        self.config = config or {}

        self.language = config.get(KEY_LANGUAGE, "en")
        self.pipeline = []

        for key, value in config.items():
            setattr(self, key, value)

    @staticmethod
    def index_the_pipeline(pipeline: List[Dict[Text, Any]]):
        if not pipeline:
            return []

        return [
            element.setdefault(ELEMENT_INDEX, index)
            for index, element in enumerate(pipeline)
        ]

    def get_element(
        self, index: int, default: Dict[Text, Any] = None
    ) -> Dict[Text, Any]:
        try:
            component = copy.deepcopy(self.pipeline[index])
            component[INDEX] = index
            return component
        except IndexError:
            if default:
                default[INDEX] = index
            else:
                default = {INDEX: index}
            return default

    def __getitem__(self, key: Text) -> Any:
        return self.config[key]
