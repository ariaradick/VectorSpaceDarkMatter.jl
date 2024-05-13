import SpecialFunctions: besselix

function normG_nli_integrand(g::GaussianF, n, ell, u, basis::RadialBasis)
    u_i = g.uSph[1]
    sigma_i = g.sigma

    z = (2*u_i*u)/sigma_i^2

    if z == 0.0
        if ell == 0
            ivefact = 4/sqrt(π)
        else
            return 0.0
        end
    else
        ivefact = sqrt(8/z) * besselix(ell+0.5, z)
    end

    measure = u^2/sigma_i^3
    expfact = exp(-(u-u_i)^2 / sigma_i^2)

    return measure * expfact * ivefact * radRn(n, ell, u, basis)
end

struct GaussianF
    c::Float64
    uSph::Vector{Float64}
    sigma::Float64
end

function scale(g::GaussianF, scale_factor)
    return GaussianF(g.c, [g.uSph[1]/scale_factor, g.uSph[2], g.uSph[3]], 
            g.sigma/scale_factor)
end

function getGnl(g::GaussianF, n, ell, radial_basis::RadialBasis;
    integ_params::NamedTuple=(;))

    umax = radial_basis.umax
    x_support = _base_of_support_n(n, radial_basis)
    gx = scale(g, umax)

    intg_Gnl(x) = normG_nli_integrand(gx, n, ell, x[1], radial_basis)

    gnl = (0.0 ± 0.0)
    for i in 1:(length(x_support)-1)
        gnl += NIntegrate(intg_Gnl,
                [x_support[i]], [x_support[i+1]], 
                :cubature; integ_params=integ_params)
    end
    return gnl
end

function getGnlm(g::GaussianF, nlm, radial_basis::RadialBasis;
    integ_params::NamedTuple=(;))

    u_i, θ_i, φ_i = g.uSph
    n, ell, m = nlm
    gnl = getGnl(g, n, ell, radial_basis; integ_params=integ_params)
    cY_i = g.c * ylm_real(ell, m, θ_i, φ_i)

    return cY_i * gnl / radial_basis.umax^3
end