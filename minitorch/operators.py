"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y


def id(x: float) -> float:
    """Return the input value unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two floats."""
    return x + y


def neg(x: float) -> float:
    """Negate a float."""
    return -x


def lt(x: float, y: float) -> float:
    """Compare two floats."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compare two floats."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floats."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two floats are close."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid of a float."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the ReLU of a float."""
    return x if x >= 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of a float."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential of a float."""
    return math.exp(x)


def log_back(x: float, g: float) -> float:
    """Calculate the derivative of the logarithm function."""
    return g / (x + EPS)


def inv(x: float) -> float:
    """Calculate the inverse of a float."""
    return 1.0 / x


def inv_back(x: float, g: float) -> float:
    """Calculate the derivative of the inverse function."""
    return -(1.0 / x**2) * g


def relu_back(x: float, g: float) -> float:
    """Calculate the derivative of the ReLU function."""
    return g if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element of a list."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to corresponding elements of two lists."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWith


def reduce(
    f: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Reduce a list to a single value using a function and an initial value."""

    def _reduce(ls: Iterable[float]) -> float:
        result = initial
        for x in ls:
            result = f(result, x)
        return result

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Use map and neg to negate each element of a list."""
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:  # Fixed typo from Interable to Iterable
    """Take the product of a list."""
    return reduce(mul, 1.0)(lst)
