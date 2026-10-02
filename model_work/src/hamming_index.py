"""Exact radius queries on 64-bit perceptual hashes; similarity is not identity."""


def checked(value):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("Expected unsigned 64-bit integer")


class HammingIndex:
    def __init__(self): self.root = None

    def add(self, value, identifier):
        checked(value)
        if self.root is None:
            self.root = [value, [identifier], {}]; return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(identifier); return
            if distance not in node[2]:
                node[2][distance] = [value, [identifier], {}]; return
            node = node[2][distance]

    def within(self, value, radius):
        checked(value)
        if type(radius) is not int or not 0 <= radius <= 64:
            raise ValueError("Expected radius 0..64")
        result = []; pending = [self.root] if self.root else []
        while pending:
            node = pending.pop(); distance = (value ^ node[0]).bit_count()
            if distance <= radius: result.extend((distance, index) for index in node[1])
            pending.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        return result
