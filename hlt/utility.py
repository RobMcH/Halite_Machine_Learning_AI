from hlt.positionals import Direction


class CyclicList(list):
    def __init__(self, length):
        self.length_ = length
        self.index_ = 0
        super(CyclicList, self).__init__()

    def append(self, obj):
        if len(self) < self.length_:
            super(CyclicList, self).append(obj)
        else:
            self[self.index_ % self.length_] = obj
            self.index_ += 1

    def check_cycle(self):
        if len(self) == 4:
            if self[0] == self[2] and self[1] == self[3] and Direction.invert(self[0]) == self[1]:
                return True
            if self[0] == self[1] == self[2] == self[3] == Direction.Still:
                return True
        return False
