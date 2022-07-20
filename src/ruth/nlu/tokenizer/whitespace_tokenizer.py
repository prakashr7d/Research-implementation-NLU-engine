from typing import List, Text

import regex
from ruth.nlu.tokenizer.tokenizer import Token, Tokenizer


class WhiteSpaceTokenizer(Tokenizer):
    def __init__(self, element_config):
        super().__init__(element_config)

    def process(self):
        pass

    def tokenize(self, text: Text) -> List[Token]:
        words = regex.sub(
            # there is a space or an end of a string after it
            r"(www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+)"
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            text,
        ).split()
        tokens = self._convert_words_to_tokens(words, text)
        return tokens
