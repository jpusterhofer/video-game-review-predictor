class Bin:
    def __init__(self):
        self.name = None
        self.lower = None
        self.upper = None
    
    def __contains__(self, key):
        if self.name is not None:
            return key == self.name
        elif self.lower is not None:
            return self.lower <= key <= self.upper
        return False
            
    def __str__(self):
        if self.name is not None:
            return self.name
        elif self.lower is not None:
            return "[" + str(self.lower) + "," + str(self.upper) + "]"