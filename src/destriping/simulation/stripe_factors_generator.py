import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import truncnorm
from scipy import stats

def normalize_stripes(stripe_factors: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize stripe factors to ensure they sum to the number of lanes.
    """
    stripe_factors_sum = np.sum(stripe_factors)
    if stripe_factors_sum > 0:
        return (stripe_factors / stripe_factors_sum) * len(stripe_factors)
    else:
        raise ValueError("Sum of stripe factors = 0")

class StripeFactors(np.ndarray):
    def __new__(cls, input_array: NDArray[np.float64]):
        # Create an instance of MyArray
        obj = np.asarray(input_array).view(cls)
        assert obj.ndim == 1
        return obj
    
    def __init__(self, input_array):
        # You can add custom initialization here if needed
        pass
    
    def plot(self,ax = None):
        return plot_stripes(self, ax)
    
    def statistics(self):
        return statistics_stripes(self)
    
    def toarray(self):
        return np.array(self)


def plot_stripes(stripe_factors: NDArray[np.float64], ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    x = np.arange(len(stripe_factors))
    ax.bar(x, stripe_factors)
    ax.set_ylabel("stripe factor")
    return ax

def statistics_stripes(stripe_factors: NDArray[np.float64]):
    mean = np.mean(stripe_factors)
    std = np.std(stripe_factors)
    sum = np.sum(stripe_factors)
    return {
        "mean": mean,
        "std": std,
        "mean/std": mean/std,
        "sum": sum
    }

class StripeFactorsGenerator():
    def __init__(self, n_lanes: int):
        self.n_lanes = n_lanes

    def generate(self)-> StripeFactors:
        pass
    

class WeibullStripeFactorsGenerator(StripeFactorsGenerator):
    def __init__(self, n_lanes: int, loc: float, shape: float, scale: float, min_val: float | None = 0.04, max_val: float|None = None, random_seed: int = None):
        super().__init__(n_lanes)
        self.min_val = min_val
        self.max_val = max_val
        self.loc = loc
        self.shape = shape
        self.scale = scale
        self.random_seed = random_seed
        self.generator = np.random.default_rng(seed=random_seed)

    def generate(self) -> StripeFactors:
        stripe_factors = stats.weibull_min.rvs(self.shape, loc=self.loc, scale=self.scale, size=self.n_lanes)
        #clip
        stripe_factors = np.clip(stripe_factors, a_min = self.min_val, a_max = self.max_val)
        return StripeFactors(stripe_factors)

class WeibullStripeFactorsGeneratorConstrained(WeibullStripeFactorsGenerator): # constrained to sum to n_lanes
    def __init__(self, n_lanes: int, loc: float, shape: float, scale: float, min_val: float| None = None, max_val: float | None = None, random_seed: int = None):
        super().__init__(n_lanes, loc, shape, scale, min_val, max_val, random_seed)

    def generate(self) -> StripeFactors:
        stripe_factors = super().generate()
        stripe_factors = normalize_stripes(stripe_factors)
        return StripeFactors(stripe_factors)