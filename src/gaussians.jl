import SpecialFunctions: besselix
import LinearAlgebra: dot

struct GaussianF
    c::Float64
    uSph::Vector{Float64}
    sigma::Float64
end

function (g::GaussianF)(uvec)
    u_cart = sph_to_cart(uvec)
    ui_cart = sph_to_cart(g.uSph)
    duvec = u_cart .- ui_cart
    du2 = dot(duvec, duvec)
    return g.c/(sqrt(π)*g.sigma)^3 * exp(-du2/g.sigma^2)
end

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

function scale(g::GaussianF, scale_factor)
    return GaussianF(g.c, [g.uSph[1]*scale_factor, g.uSph[2], g.uSph[3]], 
            g.sigma*scale_factor)
end

function getGnl(g::GaussianF, n, ell, radial_basis::RadialBasis;
    integ_params::NamedTuple=(;))

    umax = radial_basis.umax
    x_support = _base_of_support_n(n, radial_basis)
    gx = scale(g, 1/umax)

    intg_Gnl(x) = normG_nli_integrand(gx, n, ell, x[1], radial_basis)

    gnl = (0.0 ± 0.0)
    for i in 1:(length(x_support)-1)
        gnl += NIntegrate(intg_Gnl,
                [x_support[i]], [x_support[i+1]], 
                :cubature; integ_params=integ_params)
    end
    return gnl
end

function getFnlm(g::GaussianF, nlm, radial_basis::RadialBasis;
    integ_params::NamedTuple=(;))

    u_i, θ_i, φ_i = g.uSph
    n, ell, m = nlm
    gnl = getGnl(g, n, ell, radial_basis; integ_params=integ_params)
    cY_i = g.c * ylm_real(ell, m, θ_i, φ_i)

    return cY_i * gnl / radial_basis.umax^3
end

function getFnlm(g::Vector{GaussianF}, nlm, radial_basis::RadialBasis;
    integ_params::NamedTuple=(;))
    return sum(getFnlm.(g, (nlm,), (radial_basis,); integ_params=integ_params))
end

norm_energy(g::GaussianF) = g.c/(g.sigma * sqrt(2*π))^3

function norm_energy(g::Vector{GaussianF})
    res = 0.0

    for gi in g
        s2i = gi.sigma^2
        ui = sph_to_cart(gi.uSph)
        u2i = dot(ui, ui)

        for gj in g
            s2j = gj.sigma^2
            s2_ij = s2i * s2j / (s2i + s2j)

            uj = sph_to_cart(gj.uSph)
            u2j = dot(uj, uj)

            u_ij = (s2j*ui + s2i*uj) / (s2i + s2j)
            u2_ij = dot(u_ij, u_ij)

            exp_ij = exp(u2_ij/s2_ij - u2i/s2i - u2j/s2j)

            res += gi.c*gj.c*exp_ij / (π * (s2i+s2j))^(3/2)
        end
    end
    return res
end
