from typing import Dict, Text

from ruth.nlu.classifiers.hf_classifier import HFClassifier
from ruth.nlu.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from ruth.nlu.classifiers.svm_classifier import SVMClassifier
from ruth.nlu.elements import Element
from ruth.nlu.featurizers.sparse_featurizers.count_vector_featurizer import (
    CountVectorFeaturizer,
)
from ruth.nlu.featurizers.tfidf_featurizers.tfidf_vector_featurizer import (
    TfidfVectorFeaturizer,
)
from ruth.nlu.tokenizer.hf_tokenizer import HFTokenizer
from ruth.nlu.tokenizer.whitespace_tokenizer import WhiteSpaceTokenizer

element_classes = [
    # Tokenizer
    WhiteSpaceTokenizer,
    # Featurizers
    CountVectorFeaturizer,
    HFTokenizer,
    HFClassifier,
    TfidfVectorFeaturizer,
    # Classifiers
    NaiveBayesClassifier,
    SVMClassifier,
]

registered_classes: Dict[Text, Element] = {cls.name: cls for cls in element_classes}
