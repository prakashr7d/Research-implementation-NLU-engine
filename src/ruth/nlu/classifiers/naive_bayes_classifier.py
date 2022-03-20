import logging
from typing import Any, Dict, List, Text, Tuple

import sklearn
from numpy import argsort, fliplr, ndarray, reshape
from ruth.constants import INTENT, INTENT_RANKING
from ruth.nlu.classifiers import LABEL_RANKING_LIMIT
from ruth.nlu.classifiers.classifier import Classifier
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from ruth.shared.utils import json_pickle

logger = logging.getLogger(__name__)


class NaiveBayesClassifier(Classifier):
    defaults = {"priors": None, "var_smoothing": 1e-9}

    def __init__(
        self,
        element_config: Dict[Text, Any],
        le: LabelEncoder = None,
        model: "sklearn.naive_bayes.GaussianNB" = None,
    ):
        super(NaiveBayesClassifier, self).__init__(element_config=element_config)

        self.le = le or LabelEncoder()
        self.model = model

    def encode_the_str_to_int(self, labels: List[Text]) -> ndarray:
        return self.le.fit_transform(labels)

    def _create_classifier(self) -> "sklearn.naive_bayes.GaussianNB":
        from sklearn.naive_bayes import GaussianNB

        priors = self.element_config["priors"]
        var_smoothing = self.element_config["var_smoothing"]

        return GaussianNB(priors=priors, var_smoothing=var_smoothing)

    def train(self, training_data: TrainData):
        intents = [message.get(INTENT) for message in training_data.intent_examples]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intent. "
                "Atleast two unique intent are needed to train the model"
            )
            return

        X = [
            self.get_features(message).toarray()
            for message in training_data.intent_examples
        ]
        y = self.encode_the_str_to_int(intents)

        X = reshape(X, (len(X), -1))
        self.model = self._create_classifier()
        self.model.fit(X, y)

    def _predict(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        predictions = self.predict_probabilities(x)
        sorted_index = fliplr(argsort(predictions, axis=1))
        return sorted_index, predictions[:, sorted_index]

    def predict_probabilities(self, x: ndarray) -> ndarray:
        return self.model.predict_proba(x.reshape(1, -1))

    def _change_int_to_text(self, prediction: ndarray) -> ndarray:

        return self.le.inverse_transform(prediction)

    @staticmethod
    def get_features(message: RuthData) -> sparse.spmatrix:
        feature = message.get_sparse_features()
        if feature is not None:
            return feature.feature[0]
        raise ValueError("There is no sentence. Not able to train in naive bayes")

    def parse(self, message: RuthData):
        x = self.get_features(message).toarray()
        index, probabilities = self._predict(x)

        intents = self._change_int_to_text(index.flatten())
        probabilities = probabilities.flatten()

        if intents.size > 0 and probabilities.size > 0:
            ranking = list(zip(list(intents), list(probabilities)))[
                :LABEL_RANKING_LIMIT
            ]
            intent = {"name": intents[0], "accuracy": probabilities[0]}
            intent_rankings = [
                {"name": name, "accuracy": probability} for name, probability in ranking
            ]
        else:
            intent = {"name": None, "accuracy": 0.0}
            intent_rankings = []
        message.set(INTENT, intent)
        message.set(INTENT_RANKING, intent_rankings)

    def persist(self, file_name: Text, model_dir: Text):
        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name+"_encoder.pkl"

        if self.model and self.le:
            json_pickle(classifier_file_name, self.model)
            json_pickle(encoder_file_name, self.le)

        return {"file_name": file_name}

