import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Text

from rich.console import Console
from ruth.constants import TEXT
from ruth.nlu.featurizers.sparse_featurizers.constants import (
    CLASS_FEATURIZER_UNIQUE_NAME,
)
from ruth.nlu.featurizers.sparse_featurizers.sparse_featurizer import SparseFeaturizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.feature import Feature
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.utils import json_pickle, json_unpickle
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

console = Console()


class CountVectorFeaturizer(SparseFeaturizer):
    defaults = {
        "analyzer": "word",
        "stop_words": None,
        "min_df": 1,
        "max_df": 1.0,
        "min_ngram": 1,
        "max_ngram": 1,
        "lowercase": True,
        "max_features": None,
        "use_lemma": True,
    }

    def _load_params(self):
        self.analyzer = self.element_config["analyzer"]
        self.stop_words = self.element_config["stop_words"]
        self.min_df = self.element_config["min_df"]
        self.max_df = self.element_config["max_df"]
        self.min_ngram = self.element_config["min_ngram"]
        self.max_ngram = self.element_config["max_ngram"]
        self.lowercase = self.element_config["lowercase"]
        self.use_lemma = self.element_config["use_lemma"]

    def _verify_analyzer(self) -> None:
        if self.analyzer != "word":
            if self.stop_words is not None:
                logger.warning(
                    "You specified the character wise analyzer."
                    " So stop words will be ignored."
                )
            if self.max_ngram == 1:
                logger.warning(
                    "You specified the character wise analyzer"
                    " but max n-gram is set to 1."
                    " So, the vocabulary will only contain"
                    " the single characters. "
                )

    def __init__(
        self,
        element_config: Optional[Dict[Text, Any]],
        vectorizer: Optional["CountVectorizer"] = None,
    ):
        super(CountVectorFeaturizer, self).__init__(element_config)
        self.vectorizer = vectorizer
        self._load_params()
        self._verify_analyzer()

    @staticmethod
    def _build_vectorizer(
        parameters: Dict[Text, Any], vacabulary=None
    ) -> CountVectorizer:
        return CountVectorizer(
            analyzer=parameters["analyzer"],
            stop_words=parameters["stop_words"],
            min_df=parameters["min_df"],
            max_df=parameters["max_df"],
            ngram_range=(parameters["min_ngram"], parameters["max_ngram"]),
            lowercase=parameters["lowercase"],
            vocabulary=vacabulary,
        )

    def _check_attribute_vocabulary(self) -> bool:
        """Checks if trained vocabulary exists in attribute's count vectorizer."""
        try:
            return hasattr(self.vectorizer, "vocabulary_")
        except (AttributeError, KeyError):
            return False

    def create_vectors(self, examples: List[RuthData]):
        features = []
        for message in examples:
            features.append(self.vectorizer.transform([message.get(TEXT)]))
        return features

    def _get_featurizer_data(self, training_data: TrainData):
        if self._check_attribute_vocabulary():
            return self.create_vectors(training_data.training_examples)
        else:
            return []

    def _add_features_to_data(self, training_examples: List[RuthData], features):
        for message, feature in zip(training_examples, features):
            message.add_features(
                Feature(feature, self.element_config[CLASS_FEATURIZER_UNIQUE_NAME])
            )

    def train(self, training_data: TrainData) -> CountVectorizer:
        self.vectorizer = self._build_vectorizer(
            parameters={
                "analyzer": self.analyzer,
                "stop_words": self.stop_words,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "min_ngram": self.min_ngram,
                "max_ngram": self.max_ngram,
                "lowercase": self.lowercase,
            }
        )
        self.vectorizer.fit(self.get_data(training_data))
        features = self._get_featurizer_data(training_data)
        self._add_features_to_data(training_data.training_examples, features)
        return self.vectorizer

    def parse(self, message: RuthData):
        feature = self.vectorizer.transform([message.get(TEXT)])
        message.add_features(
            Feature(feature, self.element_config[CLASS_FEATURIZER_UNIQUE_NAME])
        )

    def get_vocablary_from_vectorizer(self):
        if self.vectorizer.vocabulary_:
            return self.vectorizer.vocabulary_
        else:
            raise "CountVectorizer not got trained. Please check the training data and retrain the model"

    def persist(self, file_name: Text, model_dir: Text):
        file_name = file_name + ".pkl"
        if self.vectorizer:
            vocab = self.vectorizer.vocabulary_

            featurizer_path = Path(model_dir) / file_name
            json_pickle(featurizer_path, vocab)

        return {"file_name": file_name}

    @classmethod
    def load(
        cls, meta: Dict[Text, Any], model_dir: Path, **kwargs: Any
    ) -> "CountVectorFeaturizer":
        file_name = meta.get("file_name")
        featurizer_file = model_dir / file_name

        if not featurizer_file.exists():
            return cls(meta)

        vocabulary = json_unpickle(featurizer_file)

        vectorizers = cls._build_vectorizer(parameters=meta, vacabulary=vocabulary)

        return cls(meta, vectorizers)
