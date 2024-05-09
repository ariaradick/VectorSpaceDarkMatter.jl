import SpecialFunctions: gamma

# import FastTransforms: sphevaluate
# ylm_fast(ell, m, θ, φ) = sphevaluate(θ, φ, ell, m)

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

"""
    f_uSph(f::Function; umax=1.0, z_even=false, phi_even=false, 
        phi_cyclic=1, phi_symmetric=false)

Struct that adds decorations to a function f(u, θ, φ) that tell
getFnlm various properties about the function that speed up
integration.

umax : maximum value of u for which the function is defined, used
       to scale the function to the range of the radial basis fns
z_even: (boolean) if f_uSph(x,y,z) = f_uSph(x,y,-z)
            implies <lm|f> = 0 if (l+m) is odd
phi_even: (boolean) if f_uSph(u,theta,phi) = f_uSph(u,theta,-phi)
            implies <lm|f> = 0 if m is odd
phi_cyclic: (integer) if f_uSph(u,theta,phi) = f_uSph(u,theta,phi + 2*pi/n)
phi_symmetric: (boolean) if f_uSph(u,theta) independent of phi
"""
struct f_uSph
    f::Function
    umax::Float64
    z_even::Bool
    phi_even::Bool
    phi_cyclic::Int
    phi_symmetric::Bool

    function f_uSph(f::Function; umax=1.0, z_even=false, phi_even=false, 
        phi_cyclic=1, phi_symmetric=false)
        new(f, umax, z_even, phi_even, phi_cyclic, phi_symmetric)
    end
end

abstract type RadialBasis end

struct Wavelet <: RadialBasis end

struct Tophat <: RadialBasis
    xi::Vector{Float64}
end

""" Radial basis function for spherical Haar wavelets """
function radRn(n, x, basis::Wavelet)
    haar_fn_x(n, x)
end

""" Radial basis function for spherical tophats """
function radRn(n, x, basis::Tophat)
    x_n, x_np1 = basis.xi[n+1], basis.xi[n+2]
    tophat_value(x_n, x_np1)
end

""" Full basis functions with radial basis set by radial_basis """
function phi_nlm(nlm, xvec, radial_basis::RadialBasis)
    n, ell, m = nlm
    x, θ, φ = xvec
    return radRn(n, x, radial_basis)*ylm_real(ell, m, θ, φ)
end

function _base_of_support_n(n, radial_basis::Wavelet)
    _haar_x13(n)
end

function _base_of_support_n(n, radial_basis::Tophat)
    [radial_basis.xi[n+1], radial_basis.xi[n+2]]
end

function getFnlm(f::f_uSph, nlm::Tuple{Int, Int, Int};
                 radial_basis::RadialBasis=Wavelet(),
                 integ_method::Symbol=:cubature,
                 integ_params::NamedTuple=(;))
    theta_Zn = Int(f.z_even)+1
    theta_region = [0, π/theta_Zn]

    n, l, m = nlm
    x_support = _base_of_support_n(n, radial_basis)

    if f.phi_symmetric == true
        if m ≠ 0 || (f.z_even && (l % 2 ≠ 0))
            return (0.0 ± 0.0)
        end
        function integrand_m0(x_rth)
            phi = 0.0
            xvec = [x_rth[1], x_rth[2], phi]
            uvec = [x_rth[1]*f.umax, x_rth[2], phi]
            dV_sph(xvec)*f.f(uvec)*phi_nlm(nlm, xvec, radial_basis)
        end
        fnlm = (0.0 ± 0.0)
        for i in 1:(length(x_support)-1)
            fnlm += NIntegrate(integrand_m0, 
                    [x_support[i], theta_region[1]],
                    [x_support[i+1], theta_region[2]], 
                    integ_method; integ_params=integ_params)
        end
        fnlm *= 2*π*theta_Zn
        return fnlm
    end

    if (f.z_even && ((l+m) % 2) ≠ 0) || (f.phi_even && m<0)
        return (0.0 ± 0.0)
    end

    phi_region = [0, 2*π/f.phi_cyclic]
    
    function integrand_fnlm(xvec)
        uvec = [xvec[1]*f.umax, xvec[2], xvec[3]]
        dV_sph(xvec) * f.f(uvec) * phi_nlm(nlm, xvec, radial_basis)
    end

    fnlm = (0.0 ± 0.0)
    for i in 1:(length(x_support)-1)
        fnlm += NIntegrate(integrand_fnlm, 
            [x_support[i], theta_region[1], phi_region[1]],
            [x_support[i+1], theta_region[2], phi_region[2]], 
            integ_method; integ_params=integ_params)
    end
    fnlm *= theta_Zn*f.phi_cyclic

    return fnlm
end

function getFnlm(f::Function, nlm::Tuple{Int, Int, Int};
                 radial_basis::RadialBasis=Wavelet(),
                 integ_method::Symbol=:cubature,
                 integ_params::NamedTuple=(;))
    fSph = f_uSph(f)
    return getFnlm(fSph, nlm; radial_basis=radial_basis,
        integ_method=integ_method, integ_params=integ_params)
end