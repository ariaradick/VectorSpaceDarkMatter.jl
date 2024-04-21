import SpecialFunctions: gamma
import HCubature: hcubature
import FastTransforms: sphevaluate

ylm_fast(ell, m, θ, φ) = sphevaluate(θ, φ, ell, m)

# Spherical Laguerre functions:

function laguerre(n::Int, α::Number, x::T) where {T <: Real}
    α = convert(T, α)
    p0, p1 = one(T), -x+(α+1)
    n == 0 && return p0
    for k = one(T):n-1
        p1, p0 = ((2k+α+1)/(k+1) - x/(k+1))*p1 - (k+α)/(k+1)*p0, p1
    end
    p1
end

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

struct f_uSph
    f::Function
    umax::Float64
    z_even::Bool
    phi_even::Bool
    phi_cyclic::Int
    phi_symmetric::Bool

    function f_uSph(f::Function; umax=1.0, z_even=false, phi_even=false, phi_cyclic=1,
        phi_symmetric=false)
        new(f, umax, z_even, phi_even, phi_cyclic, phi_symmetric)
    end
end

struct Wavelet end

function phi_nlm(nlm, xvec)
    n, ell, m = nlm
    x, θ, φ = xvec
    return haar_fn_x(n, x)*ylm_real(ell, m, θ, φ)
end

function getFnlm(f::f_uSph, nlm::Tuple{Int, Int, Int}; integ_params=(rtol=1e-6,
        atol=1e-6))
    theta_Zn = Int(f.z_even)+1
    theta_region = [0, π/theta_Zn]

    n, l, m = nlm
    xmin, xmid, xmax = haar_x123(n)

    if f.phi_symmetric == true
        if m ≠ 0 || (f.z_even && (l % 2 ≠ 0))
            return 0.0
        end
        function integrand_m0(x_rth)
            phi = 0.0
            xvec = [x_rth[1], x_rth[2], phi]
            uvec = [x_rth[1]*f.umax, x_rth[2], phi]
            dV_sph(xvec)*f.f(uvec)*phi_nlm(nlm, xvec)
        end
        fnlm = hcubature(integrand_m0, [xmin, theta_region[1]],
                    [xmax, theta_region[2]]; rtol=integ_params.rtol,
                    atol=integ_params.atol)
        fnlm = @. 2*π*theta_Zn*fnlm
        return fnlm
    end

    if (f.z_even && ((l+m) % 2) ≠ 0) || (f.phi_even && m<0)
        return 0.0
    end

    phi_region = [0, 2*π/f.phi_cyclic]
    
    function integrand_fnlm(xvec)
        uvec = [xvec[1]*f.umax, xvec[2], xvec[3]]
        dV_sph(xvec) * f.f(uvec) * phi_nlm(nlm, xvec)
    end

    fnlm = hcubature(integrand_fnlm, [xmin, theta_region[1], phi_region[1]],
            [xmax, theta_region[2], phi_region[2]]; rtol=integ_params.rtol,
            atol=integ_params.atol)
    fnlm = @. theta_Zn*f.phi_cyclic*fnlm

    return fnlm
end