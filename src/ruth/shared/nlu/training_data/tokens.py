from typing import Any, Dict, Text


class Tokens:
    def __init__(self, text: Text, start: int, end: int, data: Dict[Text, Any] = None):
        self.text = text
        self.start = start
        self.end = end
        self.data = data or {}

    def __eq__(self, other):
        return (self.text, self.start, self.end) == (other.text, other.start, other.end)
