import logging
from pathlib import Path
from typing import Any, Dict, List, Text, Tuple

import sklearn
from numpy import argsort, fliplr, ndarray, reshape
from rich.console import Console
from ruth.constants import INTENT, INTENT_RANKING
from ruth.nlu.classifiers import LABEL_RANKING_LIMIT
from ruth.nlu.classifiers.ruth_classifier import IntentClassifier
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.utils import json_pickle, json_unpickle
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

console = Console()


class SVMClassifier(IntentClassifier):
    defaults = {
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["auto", 0.1],
        "decision_function_shape": ["ovr"],
        "max_cross_validation_folds": 5,
        "scoring": "f1_weighted",
    }

    def __init__(
        self,
        element_config: Dict[Text, Any],
        le: LabelEncoder = None,
        clf: GridSearchCV = None,
    ):
        self.clf = clf
        super().__init__(element_config, le)

    @staticmethod
    def get_features(message: RuthData) -> sparse.spmatrix:
        feature = message.get_sparse_features()
        if feature is not None:
            return feature.feature[0]
        raise ValueError("There is no sentence. Not able to train SVMClassifier")

    @property
    def param_grids(self):
        return {
            "C": self.element_config["C"],
            "kernel": self.element_config["kernel"],
            "gamma": self.element_config["gamma"],
            "decision_function_shape": self.element_config["decision_function_shape"],
        }

    def _create_gridsearch(self, X, y) -> "sklearn.model_selection.GridSearchCV":
        from sklearn.svm import SVC

        clf = SVC(probability=True)
        param_grids = self.param_grids
        return GridSearchCV(
            clf,
            param_grids,
            scoring=self.element_config["scoring"],
            cv=self.element_config["max_cross_validation_folds"],
        )

    def train(self, training_data: TrainData):
        intents: List[Text] = [
            message.get(INTENT) for message in training_data.intent_examples
        ]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intent. "
                "At least two unique intent are needed to train the model"
            )
            return

        X = [
            self.get_features(message).toarray()
            for message in training_data.intent_examples
        ]
        y = self.encode_the_str_to_int(intents)

        X = reshape(X, (len(X), -1))
        self.clf = self._create_gridsearch(X, y)
        self.clf.fit(X, y)
        console.print(f"The Best parameter we got are {self.clf.best_params_}")
        console.print(f"score: {self.clf.best_score_}")

    def persist(self, file_name: Text, model_dir: Path):
        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name + "_encoder.pkl"

        classifier_path = model_dir / classifier_file_name
        encoder_path = model_dir / encoder_file_name

        if self.clf and self.le:
            json_pickle(classifier_path, self.clf.best_estimator_)
            json_pickle(encoder_path, self.le)

        return {"classifier": classifier_file_name, "encoder": encoder_file_name}

    def _predict(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        predictions = self.predict_probabilities(x)
        sorted_index = fliplr(argsort(predictions, axis=1))
        return sorted_index, predictions[:, sorted_index]

    def predict_probabilities(self, x: ndarray) -> ndarray:
        return self.clf.predict_proba(x.reshape(1, -1))

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Path):
        classifier_file_name = model_dir / meta["classifier"]
        encoder_file_name = model_dir / meta["encoder"]

        clf = json_unpickle(Path(classifier_file_name))
        le = json_unpickle(Path(encoder_file_name))

        return cls(meta, clf=clf, le=le)

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
