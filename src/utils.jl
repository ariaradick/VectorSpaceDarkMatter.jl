"""
    plm_norm(ell, m, x)
The 'normalized' associated Legendre polynomials.

Defined as: (-1)^m * sqrt[(l-m)! / (l+m)!] * P_lm(x)
For m=0, this is identical to the usual P_l(x).

Method:
* Using Bonnet recursion for the m=0 special case (upwards from l=0,1).
* For m>0, using 'horizontal' recursion from (m,m) to (l,m),
    using the 'associated' Bonnet recursion relations.

Numerically stable for all x in [-1,1], even arbitrarily close to x^2=1.
(e.g. x = 1 - 1e-15).
Permits the accurate calculation of P_lm(x) up to at least ell=m=1e6.
"""
function plm_norm(ell, m, x)
    # poch_plus = 1. # (l+m)!/l!
    # poch_minus = 1. # l!/(l-m)!
    Pk = zero(x)
    if ell==0
        return one(x)
    end
    x2 = x^2
    if x2 > 1
        x2 = 1.0
    end
    if x2==1
        # Evaluate now, to avoid 1/sqrt(1-x^2) division errors.
        return Int(m==0) * (x^(ell%2))
    end
    sqrt_1_x2 = sqrt(1-x2)
    if m==0
        # Upward recursion along m=0 to (l,0). Bonnet:
        if ell==1
            return x
        end
        Pk_minus2 = 1.0
        Pk_minus1 = x
        for k in 2:ell
            Pk = ((2-1/k)*x*Pk_minus1 - (1-1/k)*Pk_minus2)
            Pk_minus2 = Pk_minus1
            Pk_minus1 = Pk
        end
        return Pk
    end
        # get the (l,1) term, from the (l-1,0) and (l,0) Legendre polynomials:
    # else: use horizontal recursion from (m,m) to (ell,m)
    # modified Bonnet:
    sqrt_1_x2 = sqrt(1-x2)
    m_sqrd = 1. # l!/(l-m)! * l!/(l+m)!
    for i in 0:(m-1)
        # until i=m-1:
        m_sqrd *= 1 - 0.5/(1+i)
    end
    Pk_minus2 = sqrt_1_x2^m * sqrt(m_sqrd) #l=m
    if ell==m
        return Pk_minus2
    end
    Pk_minus1 = sqrt(2*m+1) * x * Pk_minus2 #l=m+1
    if ell==m+1
        return Pk_minus1
    end
    for k in m+2:ell
        Pk = ((2*k-1)*x*Pk_minus1 - sqrt((k-1)^2-m^2)*Pk_minus2)/sqrt(k^2-m^2)
        Pk_minus2 = Pk_minus1
        Pk_minus1 = Pk
    end
    return Pk
end

function _p_ell_all!(pk, ell_max, x)
    x2 = x^2
    if x2 >= 1.0
        # Evaluate now, to avoid 1/sqrt(1-x^2) division errors.
        return
    end

    if ell_max > 0
        pk[2] = x
    end
    for k in 2:ell_max
        pk[k+1] = ((2-1/k)*x*pk[k] - (1-1/k)*pk[k-1])
    end
end

function _plm_all!(pk, ell_max, m, x)
    x2 = x^2
    if x2 >= 1.0
        # Evaluate now, to avoid 1/sqrt(1-x^2) division errors.
        return
    end

    sqrt_1_x2 = sqrt(1-x2)
    m_sqrd = 1. # l!/(l-m)! * l!/(l+m)!
    for i in 0:(m-1)
        # until i=m-1:
        m_sqrd *= 1 - 0.5/(1+i)
    end

    pk[1] = sqrt_1_x2^m * sqrt(m_sqrd)
    if ell_max > m
        pk[2] = sqrt(2*m+1) * x * pk[1]
    end
    for k in (m+2):ell_max
        dk = k-m
        pk[dk+1] = ((2*k-1)*x*pk[dk] - sqrt((k-1)^2-m^2)*pk[dk-1])/sqrt(k^2-m^2)
    end
end

function _p_ell_all(ell_max, x)
    Pk = ones(Float64, ell_max+1)
    _p_ell_all!(Pk, ell_max, x)
    return Pk
end

function _plm_all(ell_max, m, x)
    Pk = ones(Float64, ell_max+1-m)
    if m==0
        _p_ell_all!(Pk,ell_max,x)
    else
        _plm_all!(Pk,ell_max,m,x)
    end
    return Pk
end

"Real-valued spherical harmonics."
function ylm_real(ell, m, theta, phi)
    if m > 0
        mm = m
        return ((2*ell+1)/(2*π))^0.5 * plm_norm(ell, mm, cos(theta)) * cos(mm*phi)
    elseif m < 0
        mm = -m
        return ((2*ell+1)/(2*π))^0.5 * plm_norm(ell, mm, cos(theta)) * sin(mm*phi)
    elseif m==0
        return ((2*ell+1)/(4*π))^0.5 * plm_norm(ell, m, cos(theta))
    end
    return 0.0
end

function ylm_real(lm::Tuple{Int,Int}, theta, phi)
    ell, m = lm
    return ylm_real(ell, m, theta, phi)
end

function LM_vals(lmax; z_even=false, phi_cyclic=1, phi_even=false, 
                 phi_symmetric=false, center_Z2=false)
    z = Int(z_even) + 1
    cz2 = center_Z2 + 1
    if phi_symmetric
        if z_even
            return [(ell,0) for ell in 0:z:lmax]
        elseif center_Z2
            return [(ell,0) for ell in 0:cz2:lmax]
        else
            return [(ell,0) for ell in 0:lmax]
        end
    elseif phi_even
        return [(ell,m) for ell in 0:cz2:lmax for m in (ell%z):z:ell if (m%phi_cyclic==0)]
    else
        return [(ell,m) for ell in 0:cz2:lmax for m in -ell:z:ell if (m%phi_cyclic==0)]
    end
end

"Gives the `(l,m)` that correspond to a given `i` in a spherical harmonic vector."
function LM_sph(i)
    l = floor(Int, sqrt(i-1))
    m = i - (l^2+l+1)
    return (l,m)
end

"Gives the `i` that corresponds to `ell` and `m` for a spherical harmonic vector."
function idx_sph(ell, m)
    ell^2+ell+1+m
end

"Integration measure in 3D spherical coordinates."
function dV_sph(rvec)
    return rvec[1]^2 * sin(rvec[2])
end

"Thin wrapper for HCubature and MCIntegration that uses HCubature's syntax."
function NIntegrate(integrand::Function, a::Vector{Float64}, 
    b::Vector{Float64}, method::Symbol; integ_params::NamedTuple=(;))

    if method == :cubature
        if !(:rtol in keys(integ_params))
            intg_params = (rtol=1e-6, integ_params...)
        else
            intg_params = integ_params
        end
        return measurement(hcubature(integrand, a, b; intg_params...)...)

    elseif method in (:vegas, :vegasmc)
        function intg(mcvec, c)
            xvec = [y[1] for y in mcvec]
            integrand(xvec)
        end
        res = integrate( intg;
                solver=method,
                var=Continuous([(a[i],b[i]) for i in eachindex(a)]),
                integ_params... )
        return measurement(res.mean[1], res.stdev[1])

    else
        error("Integration method $method not supported.")
    end
end

function NIntegrate(integrand::Function, a, b; integ_params::NamedTuple=(;))
    NIntegrate(integrand, a, b, :cubature; integ_params=integ_params)
end

"Converts a vector in spherical coordinates to cartestian coordinates."
function sph_to_cart(x_sph)
    x, θ, φ = x_sph
    rx = x*sin(θ)*cos(φ)
    ry = x*sin(θ)*sin(φ)
    rz = x*cos(θ)
    return [rx, ry, rz]
end

"Converts a vector in cartestian coordinates to spherical coordinates."
function cart_to_sph(uXYZ)
    ux, uy, uz = uXYZ
    u = sqrt(ux^2 + uy^2 + uz^2)
    uxy = sqrt(ux^2 + uy^2)
    # first address uxy=0 and ux=0 special cases...
    phi = 0 #arbitrary; phi not well defined at theta=0,pi
    if uxy==0
        if uz >= 0
            theta = 0
        else
            theta = π
        end
        return [u, theta, phi]
    end
    theta = 0.5*pi - atan(uz/uxy)
    # ux=0...
    if ux == 0
        if uy > 0
            phi = 0.5*pi
        elseif uy < 0
            phi = 1.5*pi
        end
        return [u, theta, phi]
    end
    # Now, non-special cases...
    if ux > 0 && uy > 0
        phi = atan(uy/ux)
    elseif ux < 0 && uy < 0
        phi = atan(uy/ux) + pi
    elseif ux < 0 && uy > 0
        phi = atan(uy/ux) + pi
    elseif ux > 0 && uy < 0
        phi = atan(uy/ux) + 2*pi
    end
    return [u, theta, phi]
end

function _fsht_to_vsdm(C,ellmax)
    lm_vals = LM_vals(ellmax)
    res = zeros(Float64, length(lm_vals))
    for i in eachindex(lm_vals)
        res[i] = C[sph_mode(lm_vals[i]...)]
    end
    return res
end