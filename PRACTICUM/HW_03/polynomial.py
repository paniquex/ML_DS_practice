class Polynomial:
    def __init__(self, *args):
        if len(args) == 0:
            self.coefs = 0
        else:
            self.coefs = [coef for coef in args]

    def __call__(self, x):
        if self.coefs == 0:
            return 0
        return sum([x ** idx * coef for idx, coef in enumerate(self.coefs)])
