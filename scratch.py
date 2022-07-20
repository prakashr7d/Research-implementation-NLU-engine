from typing import Text, Dict, Any


class Tokens:
    def __init__(self, text: Text, start: int, end: int, data: Dict[Text, Any] = None):
        self.text = text
        self.start = start
        self.end = end
        self.data = data or {}

    def __eq__(self, other):
        return (self.text, self.start, self.end) == (other.text, other.start, other.end)


object1 = Tokens("Hai this is sharu", 1, 2)
object2 = Tokens("Hai this is sharu", 2, 3)
print(object1 == object2)
