#!/usr/bin/env python3.9
from typing import Callable, Tuple, Dict, List, TypeVar
from sympy import Symbol, Basic, LC as leading_coeff, roots, together, denom, numer, Poly

# most of these were taken from an old repo of mine
# https://github.com/fakuivan/facu-ciran/blob/master/utils.py

def f2nd(function: Basic) -> Tuple[Basic, Basic]:
    function = together(function)
    return numer(function), denom(function)

def f2zpk(function: Basic, var: Symbol
) -> Tuple[Dict[Basic, int], Dict[Basic, int], Basic]:
    numer, denom = f2nd(function)
    numer_lc, denom_lc = map(
        lambda expr: leading_coeff(expr, var),
        (numer, denom))
    # It's not really necesary to devide by the leading
    # coefficient, but what do I know
    return roots(numer/numer_lc, var), \
           roots(denom/denom_lc, var), numer_lc/denom_lc

def ratpoly_coeffs(function: Basic, var: Symbol
) -> Tuple[List[Basic], List[Basic]]:
    numer, denom = f2nd(function)
    return (Poly(numer, var).coeffs(),
            Poly(denom, var).coeffs())
