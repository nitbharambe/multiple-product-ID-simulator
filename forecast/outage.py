from scipy.stats import bernoulli


class Outage:
    def __init__(self, out_probability, out_percentage):
        assert out_probability <= 1. and out_probability >= 0, "Probability out of range"
        assert out_percentage <= 1. and out_percentage >= 0, "Out part out of range"

        self.dist = bernoulli(out_probability)
        self.out_part = out_percentage
        self.status = False

    def update_capacity(self, capacity):
        s = self.dist.rvs() > 0
        new_capacity = self.out_part * capacity if (not self.status and s) else capacity
        self.status = s or self.status
        return new_capacity
