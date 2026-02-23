import numpy as np

class RunningStat:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)

    def variance(self):
        return self.S / (self.n - 1) if self.n > 1 else np.ones_like(self.mean)

    def std(self):
        return np.sqrt(self.variance())