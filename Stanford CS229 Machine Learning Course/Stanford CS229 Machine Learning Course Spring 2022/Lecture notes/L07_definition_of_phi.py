import numpy as np
from itertools import product


def generate_monomials(x):
    """
    Generate all monomials up to the third degree for a 3D vector x.

    Args:
    x (list or np.ndarray): A 3D vector [x1, x2, x3]

    Returns:
    np.ndarray: Array containing all the monomials including repetitions.
    """
    x = np.asarray(x)
    d = len(x)
    max_degree = 3

    # Generate monomials for each degree
    monomials = []
    for degree in range(max_degree + 1):
        for exponents in product(range(d), repeat=degree):
            if not exponents:
                monomial = 1
            else:
                monomial = np.prod([x[i] for i in exponents])
            monomials.append(monomial)

    return np.array(monomials)


# Example usage
x = [1, 2, 3]
monomials = generate_monomials(x)
print(monomials)
print(f"Total monomials: {len(monomials)}")
