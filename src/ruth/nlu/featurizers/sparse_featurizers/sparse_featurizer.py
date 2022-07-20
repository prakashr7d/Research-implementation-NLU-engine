from typing import Any, Dict, List, Text

from ruth.nlu.featurizers.sparse_featurizers.constants import (
    CLASS_FEATURIZER_UNIQUE_NAME,
)
from ruth.shared.nlu.ruth_elements import Element
from ruth.shared.nlu.training_data.collections import TrainData


class SparseFeaturizer(Element):
    def __init__(self, element_config):
        element_config = element_config or {}
        self.element_config = element_config
        element_config.setdefault(
            CLASS_FEATURIZER_UNIQUE_NAME, self.create_unique_name()
        )
        super().__init__(element_config)

    def _build_vectorizer(self, parameters: Dict[Text, Any]):
        raise NotImplementedError

    @staticmethod
    def get_data(training_data: TrainData) -> List[Text]:
        return training_data.get_text_list(
            training_examples=training_data.training_examples
        )
