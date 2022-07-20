import logging
from abc import ABC
from typing import Any, Dict, List, Text, Tuple

import torch
from numpy import argsort, fliplr, ndarray
from ruth.constants import INTENT, INTENT_RANKING
from ruth.nlu.classifiers import LABEL_RANKING_LIMIT
from ruth.nlu.classifiers.constants import BATCH_SIZE, EPOCHS, MODEL_NAME
from ruth.nlu.classifiers.ruth_classifier import IntentClassifier
from ruth.shared.constants import TOKENS
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)

set_seed(42)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    logger.debug("No GPU found!, Using CPU instead")
    device = torch.device("cpu")


def train_model(dataloader, optimizer_, scheduler_, device_):
    global model

    predictions_labels = []
    true_labels = []

    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch["labels"].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, device_):
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch["labels"].numpy().flatten().tolist()

        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


def _create_classifier(n_labels):
    model_config_ = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, num_labels=n_labels
    )
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, config=model_config_
    )
    return transformer_model


def get_tokens(message: RuthData) -> List[torch.Tensor]:
    return message.get(TOKENS)


class CreateData(Dataset, ABC):
    def __init__(self, training_data: TrainData, le: LabelEncoder = None):
        self.training_data = training_data
        intents = [message.get(INTENT) for message in training_data.intent_examples]
        if len(set(intents)) < 2:
            logger.warning(
                "There are no enough intents"
                "At least two unique intents are needed to train the model"
            )
            return
        self.labels = intents
        self.le = le or LabelEncoder()
        self.encoder = le.fit(intents)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, item: int):
        data = self.training_data[item]
        token_ids = data.get(TOKENS)
        label = data.get(INTENT)
        label_encoded = self.encoder.transform(label)
        return token_ids, label_encoded


class BertClassifier(IntentClassifier):
    def __init__(self, element_config: Dict[Text, Any] = None, le: LabelEncoder = None):
        super(BertClassifier, self).__init__(element_config=element_config)
        self.le = le or LabelEncoder()
        self.optimizer = AdamW(
            model.parameters(),
            lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
        )
        self.total_steps = len(self.train_dataloader) * EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

    def encode_the_str_to_int(self, labels: List[Text]) -> ndarray:
        return self.le.fit_transform(labels)

    def train(self, training_data: TrainData):
        all_loss = {"train_loss": [], "val_loss": []}
        all_acc = {"train_acc": [], "val_acc": []}

        for epoch in tqdm(range(EPOCHS)):
            train_labels, train_predict, train_loss = train_model(
                self.train_dataloader, self.optimizer, self.scheduler, device
            )
            train_acc = accuracy_score(train_labels, train_predict)

            valid_labels, valid_predict, val_loss = validation(
                self.valid_dataloader, device
            )
            val_acc = accuracy_score(valid_labels, valid_predict)

            all_loss["train_loss"].append(train_loss)
            all_loss["val_loss"].append(val_loss)
            all_acc["train_acc"].append(train_acc)
            all_acc["val_acc"].append(val_acc)

    def _loader(self, training_data: TrainData, valid_data: TrainData):
        train_dataset = CreateData(training_data, self.le)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        valid_dataset = CreateData(valid_data)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

    @staticmethod
    def _predict(x: torch.Tensor) -> Tuple[ndarray, ndarray]:
        predictions = model(x)
        sorted_index = fliplr(argsort(predictions, axis=1))
        return sorted_index, predictions[:, sorted_index]

    def _change_int_to_text(self, prediction: ndarray) -> ndarray:
        return self.le.inverse_transform(prediction)

    def parse(self, message: RuthData):
        X = message.get(TOKENS)
        index, probabilities = self._predict(X)

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
