import SpecialFunctions: besselix

function normG_nli_integrand(radR_nlu, u_i, sigma_i, n, ell, u)
    z = (2*u_i*u)/sigma_i^2

    if z == 0
        if ell == 0
            ivefact = 4/sqrt(π)
        else
            return zero(u)
        end
    else
        ivefact = sqrt(8/z) * besselix(ell+0.5, z)
    end

    measure = u^2/sigma_i^3
    expfact = exp(-(u-u_i)^2 / sigma_i^2)

    return measure * expfact * radR_nlu(n, ell, u)
end

struct GaussianF
    c::Vector{Float64}
    uSph::Vector{Vector{Float64}}
    sigma::Vector{Float64}
end

function GaussianF(c::Real, uSph::Vector, sigma::Real)
    GaussianF([Float64(c)], [uSph], [Float64(sigma)])
end