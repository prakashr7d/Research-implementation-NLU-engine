from typing import List, Dict, Text, Any, Optional, Tuple

import torch
from transformers import BertTokenizer

from ruth.nlu.constants import ELEMENT_UNIQUE_NAME
from ruth.nlu.tokenizer.constants import max_length_for_padding
from ruth.nlu.tokenizer.tokenizer import Tokenizer
from ruth.shared.nlu.training_data.collections import TrainData
from ruth.shared.nlu.training_data.ruth_data import RuthData
from ruth.shared.nlu.training_data.tokens import Tokens


class TokenizerBert(Tokenizer):

    def __init__(self, element_config: Optional[Dict[Text, Any]] = None, tokenizer: Optional["BertTokenizer"] = None):
        super(TokenizerBert, self).__init__(element_config)
        self.tokenizer = tokenizer or {}

    def _build_tokenizer(self,
                         # parameters: Dict[Text, Any]
                         ) -> BertTokenizer:
        return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def create_tokens(self, examples: List[RuthData]):
        tokens = []
        attention_masks = []
        for message in examples:
            encoded_dict = self.tokenizer.encode_plus(message.text,
                                                      add_special_tokens=True,
                                                      max_length=max_length_for_padding,
                                                      padding=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')

            tokens.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        return torch.cat(tokens, dim=0), torch.cat(attention_masks, dim=0)

    def _get_tokenized_data(self, training_data: TrainData):
        return self.create_tokens(training_data.training_examples)

    def _add_tokens_to_data(self,
                            training_examples: List[RuthData],
                            tokens: torch.Tensor,
                            attention_masks: torch.Tensor):
        for message, token, attention_mask in zip(training_examples, tokens, attention_masks):
            message.add_tokens(Tokens(token, self.element_config[ELEMENT_UNIQUE_NAME]))
            message.add_attention_masks(Tokens(attention_mask, self.element_config[ELEMENT_UNIQUE_NAME]))

    def train(self, training_data: TrainData) -> BertTokenizer:
        self.tokenizer = self._build_tokenizer()  # TODO: add parameters
        tokens, attention_masks = self._get_tokenized_data(training_data)
        self._add_tokens_to_data(training_data.training_examples, tokens, attention_masks)
        return self.tokenizer

    def parse(self, message: RuthData) -> Tuple[torch.Tensor, torch.Tensor]:
        parse_encoded = self.tokenizer.encode_plus(message.text,
                                                   add_special_tokens=True,
                                                   max_length=max_length_for_padding,
                                                   padding=True,
                                                   return_attention_mask=True,
                                                   return_tensors='pt')

        token_ids = parse_encoded['input_ids']
        attentions_masks = parse_encoded['attention_masks']


