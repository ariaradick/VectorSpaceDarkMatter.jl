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
    f_uSph(f::Function; z_even=false, phi_even=false, 
        phi_cyclic=1, phi_symmetric=false)

Struct that adds decorations to a function f(u, θ, φ) that tell
getFnlm various properties about the function that speed up
integration.

z_even: (boolean) if f_uSph(x,y,z) = f_uSph(x,y,-z)
            implies <lm|f> = 0 if (l+m) is odd
phi_even: (boolean) if f_uSph(u,theta,phi) = f_uSph(u,theta,-phi)
            implies <lm|f> = 0 if m is odd
phi_cyclic: (integer) if f_uSph(u,theta,phi) = f_uSph(u,theta,phi + 2*pi/n)
phi_symmetric: (boolean) if f_uSph(u,theta) independent of phi
"""
struct f_uSph
    f::Function
    z_even::Bool
    phi_even::Bool
    phi_cyclic::Int
    phi_symmetric::Bool

    function f_uSph(f::Function; z_even=false, phi_even=false, 
        phi_cyclic=1, phi_symmetric=false)
        new(f, z_even, phi_even, phi_cyclic, phi_symmetric)
    end
end

function LM_vals(f::f_uSph, lmax)
    LM_vals(lmax; z_even=f.z_even, phi_even=f.phi_even, 
            phi_symmetric=f.phi_symmetric)
end

abstract type RadialBasis end

"""
    Wavelet(umax)

Spherical Haar wavelets. Contains the maximum value of u = \$|\\vec{u}|\$ that the 
basis of will be evaluated over.
"""
struct Wavelet <: RadialBasis
    umax::Float64
end

Wavelet() = Wavelet(1.0)

"""
    Tophat(xi, [umax])

Tophat basis functions between each point in xi. The range will be scaled by
umax, the maximum value of u = |\\vec{u}|.
"""
struct Tophat <: RadialBasis
    xi::Vector{Float64}
    umax::Float64
end

Tophat(xi) = Tophat(xi, 1.0)

""" Radial basis function for spherical Haar wavelets """
function radRn(n, ell, u, basis::Wavelet)
    haar_fn_x(n, u/basis.umax)
end

""" Radial basis function for spherical tophats """
function radRn(n, ell, u, basis::Tophat)
    x = u/basis.umax
    x_n, x_np1 = basis.xi[n+1], basis.xi[n+2]

    if (x_n < x < x_np1)
        return tophat_value(x_n, x_np1)
    elseif x ≈ x_n
        if x ≈ 0.0
            return tophat_value(x_n, x_np1)
        else
            return 0.5*tophat_value(x_n, x_np1)
        end
    elseif x ≈ x_np1
        if x ≈ 1.0
            return tophat_value(x_n, x_np1)
        else
            return 0.5*tophat_value(x_n, x_np1)
        end
    else
        return 0.0
    end
end

""" Full basis functions with radial basis set by radial_basis """
function phi_nlm(nlm, uvec, radial_basis::RadialBasis)
    n, ell, m = nlm
    u, θ, φ = uvec
    return radRn(n, ell, u, radial_basis)*ylm_real(ell, m, θ, φ)
end

phi_nlm(n, lm, uvec, radial_basis::RadialBasis) = phi_nlm((n, lm...), uvec, radial_basis)

"Returns the base of support (x1, x2) for a given n and radial basis"
function _base_of_support_n(n, radial_basis::Wavelet)
    haar_x123(n)
end

function _base_of_support_n(n, radial_basis::Tophat)
    [radial_basis.xi[n+1], radial_basis.xi[n+2]]
end

"""
    getFnlm(f, nlm::Tuple{Int, Int, Int},
        radial_basis::RadialBasis;
        integ_method::Symbol=:cubature,
        integ_params::NamedTuple=(;))

Calculates the (n,l,m) coefficient <f | nlm> for an expansion in the specified
radial basis (up to radial_basis.umax) and real spherical harmonics.

`f` : Can be a `Function`, `f_uSph`, or `GaussianF`. `f_uSph` is preferred if
      your function has any symmetries, as specifying those will greatly speed 
      up evaluation.

`radial_basis` : Either a `Wavelet` or `Tophat`

`integ_method` : can be `:cubature`, `:vegas`, or `:vegasmc`

`integ_params` : keyword arguments to pass to the integrator. If `:cubature`, 
    these are kwargs for `hcubature`. If `:vegas` or `:vegasmc`, these are
    kwargs for `MCIntegration`'s `integrate` method.
"""
function getFnlm(f::f_uSph, nlm::Tuple{Int, Int, Int},
                 radial_basis::RadialBasis;
                 integ_method::Symbol=:cubature,
                 integ_params::NamedTuple=(;))
    theta_Zn = Int(f.z_even)+1
    theta_region = [0, π/theta_Zn]

    n, l, m = nlm
    x_support = _base_of_support_n(n, radial_basis)
    u_max = radial_basis.umax

    if f.phi_symmetric == true
        if m ≠ 0 || (f.z_even && (l % 2 ≠ 0))
            return (0.0 ± 0.0)
        end
        function integrand_m0(x_rth)
            phi = 0.0
            xvec = [x_rth[1], x_rth[2], phi]
            uvec = [x_rth[1]*u_max, x_rth[2], phi]
            dV_sph(xvec)*f.f(uvec)*phi_nlm(nlm, uvec, radial_basis)
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
        uvec = [xvec[1]*u_max, xvec[2], xvec[3]]
        dV_sph(xvec) * f.f(uvec) * phi_nlm(nlm, uvec, radial_basis)
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

function getFnlm(f::Function, nlm::Tuple{Int, Int, Int},
                 radial_basis::RadialBasis;
                 integ_method::Symbol=:cubature,
                 integ_params::NamedTuple=(;))
    fSph = f_uSph(f)
    return getFnlm(fSph, nlm, radial_basis;
        integ_method=integ_method, integ_params=integ_params)
end

"""
    getFnlm(f, nlm::Tuple{Int, Int, Int}; umax=1.0,
        integ_method::Symbol=:cubature, integ_params::NamedTuple=(;))

If called without a `radial_basis` argument, will automatically use `Wavelet`
with umax as a keyword argument.
"""
function getFnlm(f::f_uSph, nlm::Tuple{Int, Int, Int}; umax=1.0,
    integ_method::Symbol=:cubature, integ_params::NamedTuple=(;))
    radial_basis = Wavelet(umax)
    return getFnlm(f, nlm, radial_basis;
        integ_method=integ_method, integ_params=integ_params)
end

function getFnlm(f::Function, nlm::Tuple{Int, Int, Int}; umax=1.0,
    integ_method::Symbol=:cubature, integ_params::NamedTuple=(;))
    fSph = f_uSph(f)
    radial_basis = Wavelet(umax)
    return getFnlm(fSph, nlm, radial_basis;
        integ_method=integ_method, integ_params=integ_params)
end