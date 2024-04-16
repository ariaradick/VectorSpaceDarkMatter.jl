import SpecialFunctions: gamma
import ClassicalOrthogonalPolynomials: laguerrel as laguerre

include("haar.jl")

# Spherical Laguerre functions:

_lag_half_ln(n, ell, z) = laguerre(n, 1/2 + ell, z)

"""Normalized spherical Laguerre function"""
function lag_spherical(n, ell, x)
    factor = sqrt(factorial(n) * 2^(2.5+ell) / gamma(n+1.5+ell))
    return factor * exp(-x^2) * x^ell * _lag_half_ln(n, ell, 2*x^2)
end

# Tophat:

"""
    tophat_value(x1, x2; dim=3)

Returns the value of the tophat function where it is nonzero (x1 < x < x2),
assuming x1 ≠ x2 and both are non-negative.
Normalized by `int(x^2 dx * value^2, {x, x1, x2}) = 1`.
"""
tophat_value(x1, x2; dim=3) = sqrt(dim / (x2^dim - x1^dim))

