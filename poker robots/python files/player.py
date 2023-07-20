class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.opponent = False
        self.hand = []
        self.action = []