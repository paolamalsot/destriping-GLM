from typing import Any
from glum._distribution import NegativeBinomialDistribution
import xxhash

def set_family_arg(theta):
    # potential way to extend to other families:
    # family_arg = self.regressor_args["family"]
    # family = get_family(family_arg)
    # family.theta = theta
    family = NegativeBinomialDistribution(theta)
    return family


def family_is_negative_binomial(family: Any) -> bool:
    return isinstance(family, NegativeBinomialDistribution) or (family == "negative.binomial")


# HELPERS

def hash(coef):
    return xxhash.xxh64(coef, seed=0).hexdigest()