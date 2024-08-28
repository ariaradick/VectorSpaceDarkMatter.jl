function _b_nk_int(n::Int, k::Int, x)
    res = 0.0
    for j in 0:k
        comb = binomial(Float64(k),j)
        ipower = j - n - k/2 + 1
        if ipower ≈ 0
            res += comb*log(x)
        else
            res += comb * x^ipower / ipower
        end
    end
    return 0.5 * res
end

function _c_alpha_int(α, x)
    if α == -2
        return -0.5*log( (1+x)/x ) + 0.5/x - 0.125/x^2
    elseif α == -1
        return log((1+x)/x) - 0.5/x
    elseif α == 0
        return (-0.25*log(x)^2 + log(x)*log(1+x) + reli2(-x))
    elseif α == 1
        return 0.5*x - log(1+x)
    elseif α == 2
        return 0.125*x^2 - 0.5*x + 0.5*log(1+x)
    else
        return x^α / α^2 * (0.5 - hyp2f1(1, α, 1+α, -x))
    end
end

function _v_n_int(n, x)
    if n==0
        return 0.5*log(x) + x + 0.25*x^2
    elseif n==1
        return 0.5/x + log(x) + 0.5*x
    elseif n==2
        return -0.25/x^2 - 1/x + 0.5*log(x)
    else
        return 0.5*(x^(-n)/(-n)) + x^(-n+1)/(-n+1) + 0.5*x^(-n+2)/(-n+2)
    end
end

function _s_n_int(n, x)
    logfactor = 0.5*log(x/(1+2*x+x^2))
    return (logfactor*_v_n_int(n, x) + 0.5*_c_alpha_int(-n, x)
            + _c_alpha_int(1-n, x) + 0.5*_c_alpha_int(2-n, x))
end

"""
    t_ln_vq_int(l::Integer, n, v12_star, q12_star)

Rectangular integral T_{l,n}.

With [v1,v2] in units of v_star = q_star/mX, [q1,q2] in units of
    q_star = sqrt(2*mX*omegaS).

Always v1 >= 1. Also require q1 > 0.
"""
function t_ln_vq_int(l::Integer, n, v12_star, q12_star)
    v1,v2 = v12_star
    q1,q2 = q12_star
    x1 = q1^2
    x2 = q2^2
    res = 0.0
    for k in (l%2):2:l
        term_k = 2^(l-k) * gamma(0.5*(k+1+l)) / gamma(0.5*(k+1-l))
        if k==2
            termV = log(v2/v1) / (2 * gamma(l-1))
            termQ = (_b_nk_int(n, 2, x2) - _b_nk_int(n, 2, x1))
            res += termV*term_k*termQ
        else
            termV = ((v2^(2-k) - v1^(2-k))
                     / ((2-k)*gamma(k+1)*gamma(l-k+1)))
            termQ = (_b_nk_int(n, k, x2) - _b_nk_int(n, k, x1))
            res += termV*term_k*termQ
        end
    end
    return res
end

"""
    u_ln_vq_int(l::Integer, n, v2_star, q12_star)

Non-rectangular integral U_{l,n}, with lower bound v1 = v_min(q).

With v2 in units of v_star = q_star/mX, [q1,q2] in units of
    q_star = sqrt(2*mX*omegaS).
"""
function u_ln_vq_int(l::Integer, n, v2_star, q12_star)
    v2 = v2_star # only need v2, v1 is irrelevant
    q1,q2 = q12_star
    x1 = q1^2
    x2 = q2^2
    sum = 0.0
    for k in (l%2):2:l
        term_k = (gamma(0.5*(k+1+l)) / gamma(0.5*(k+1-l))
                  * 2.0^(l-k)/(gamma(k+1) * gamma(l-k+1)))
        if k==2
            summand = (log(2.0*v2)*_b_nk_int(n, 2, x2) + _s_n_int(n, x2)
                       - log(2.0*v2)*_b_nk_int(n, 2, x1) - _s_n_int(n, x1))
            sum += term_k*summand
        else
            summand = (v2^(2-k)*(_b_nk_int(n, k, x2) - _b_nk_int(n, k, x1))
                       - 2.0^(k-2)*(_b_nk_int(n, 2, x2) - _b_nk_int(n, 2, x1)))
            sum += term_k*summand/(2-k)
        end
    end
    return sum
end

"""
    mI_star(ell, n, v12_star, q12_star)

Dimensionless integral related to MathcalI.

This is 'I^{(\\ell)}_\\star' without the prefactor (qBohr/qStar)**(2n)

With v12, q12 in units of vStar, qStar.

There are 0, 1, 2 or 3 regions that contribute to mcalI:
    qA < (R1) < qB < (R2) < qC < (R3) < qD.
    R2 is rectangular, bounded by v1 < v < v2. -> t_ln_vq_int
    R1 and R3 are not rectangular: vMin(q) < v < v2. -> u_ln_vq_int
If vmin(q) > v1 for all q1 < q < q2, then mcalI is given by u_ln_vq_int
"""
function mI_star(ell, n, v12_star, q12_star)
    v1,v2 = v12_star
    q1,q2 = q12_star
    if v1==v2 || q1==q2
        return 0.0 # No integration volume
    end
    if q2 < q1
        error("q12 must be ordered.")
    end
    if v2 < v1
        error("v12 must be ordered.")
    end
    include_R2 = true
    if v2 < 1.0
        # v2 is below the velocity threshold. mcaI=0
        return 0.0
    end
    tilq_m,tilq_p = v2 - sqrt(v2^2-1.0), v2 + sqrt(v2^2-1.0)
    if tilq_m > q2 || tilq_p < q1
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return 0.0
    end
    # Else: there are some q satisfying vmin(q) < v2 in this interval.
    if v1 < 1.0
        # There is no v1 = vmin(q) solution for any real q
        include_R2 = false
    # Else: There are two real solutions to v1 = vmin(q)
    else
        q_m,q_p = v1 - sqrt(v1^2-1.0), v1 + sqrt(v1^2-1.0)
        if q_m > q2 || q_p < q1
            # in this case v1 < vmin(q) for all q in [q1,q2]
            include_R2 = false
        end
    end
    if !include_R2
        q_A = max(q1, tilq_m)
        q_B = min(q2, tilq_p)
        return u_ln_vq_int(ell, n, v2, [q_A, q_B])
    end
    # Else: at least part of the integration volume is set by v1 < v.
    q_a = max(q1, tilq_m)
    q_b = max(q1, q_m) # q_m > tilq_m iff v2 > v1
    q_c = min(q2, q_p) # q_p < tilq_p iff v2 > v1
    q_d = min(q2, tilq_p)

    includeRegion = [true, true, true]
    if q_a==q_b
        includeRegion[1] = false
    end

    if q_c==q_d
        includeRegion[3] = false
    end

    if v1>1.0
        if q_b == q_c
            error("If q_b==q_c then there should be no R2 region...")
        end
    end
    mI_0,mI_1,mI_2 = 0.0, 0.0, 0.0
    if includeRegion[1]
        mI_0 = u_ln_vq_int(ell, n, v2, [q_a, q_b])
    end
    if includeRegion[2]
        mI_1 = t_ln_vq_int(ell, n, [v1,v2], [q_b, q_c])
    end
    if includeRegion[3]
        mI_2 = u_ln_vq_int(ell, n, v2, [q_c, q_d])
    end
    return mI_0 + mI_1 + mI_2
end

function T_matrix(ℓmax)
    res = zeros(Float64, (ℓmax+1, ℓmax+1))
    for ell in 0:ℓmax
        for k in (ell%2):2:ell
            res[k+1,ell+1] = 2^(ell-k)*gamma(0.5*(k+ell+1)) / (gamma(k+1) *
                             gamma(ell-k+1) * gamma(0.5*(k-ell+1)))
        end
    end
    return LowerTriangular(res')
end

function add_T_vector!(I, a, v1, v2, q1, q2; fdmn=0)
    for i in eachindex(I)
        if i==3
            I[i] += a*(VSDM._b_nk_int(fdmn, 2, q2^2)-
                     VSDM._b_nk_int(fdmn, 2, q1^2))*log(v2/v1)
        else
            k = i-1
            I[i] += a*(VSDM._b_nk_int(fdmn, k, q2^2)-
                    VSDM._b_nk_int(fdmn, k, q1^2)) * 
                    (v2^(2-k) - v1^(2-k)) / (2-k)
        end
    end
end

function add_U_vector!(I, a, v2, q1, q2; fdmn=0)
    dB2 = VSDM._b_nk_int(fdmn,2,q2^2) - VSDM._b_nk_int(fdmn,2,q1^2)
    for i in eachindex(I)
        if i==3
            I[i] += a*(log(2*v2)*dB2 + VSDM._s_n_int(fdmn,q2^2) - 
                    VSDM._s_n_int(fdmn,q1^2))
        else
            k = i-1
            I[i] += a*(v2^(2-k) / (2-k) * (VSDM._b_nk_int(fdmn,k,q2^2) - 
                    VSDM._b_nk_int(fdmn,k,q1^2)) - 2.0^(k-2)*dB2/ (2-k))
        end
    end
end

function not_mI_star!(I, a, n, v12_star, q12_star)
    v1,v2 = v12_star
    q1,q2 = q12_star
    if v1==v2 || q1==q2
        return 0.0 # No integration volume
    end
    if q2 < q1
        error("q12 must be ordered.")
    end
    if v2 < v1
        error("v12 must be ordered.")
    end
    include_R2 = true
    if v2 < 1.0
        # v2 is below the velocity threshold. mcaI=0
        return 0.0
    end
    tilq_m,tilq_p = v2 - sqrt(v2^2-1.0), v2 + sqrt(v2^2-1.0)
    if tilq_m > q2 || tilq_p < q1
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return 0.0
    end
    # Else: there are some q satisfying vmin(q) < vr2 in this interval.
    if v1 < 1.0
        # There is no v1 = vmin(q) solution for any real q
        include_R2 = false
    # Else: There are two real solutions to v1 = vmin(q)
    else
        q_m,q_p = v1 - sqrt(v1^2-1.0), v1 + sqrt(v1^2-1.0)
        if q_m > q2 || q_p < q1
            # in this case v1 < vmin(q) for all q in [q1,q2]
            include_R2 = false
        end
    end
    if !include_R2
        q_A = max(q1, tilq_m)
        q_B = min(q2, tilq_p)
        add_U_vector!(I, a, v2, q_A, q_B; fdmn=n)
    else
        # Else: at least part of the integration volume is set by v1 < v.
        q_a = max(q1, tilq_m)
        q_b = max(q1, q_m) # q_m > tilq_m iff v2 > v1
        q_c = min(q2, q_p) # q_p < tilq_p iff v2 > v1
        q_d = min(q2, tilq_p)

        if v1>1.0
            if q_b == q_c
                error("If q_b==q_c then there should be no R2 region...")
            end
        end

        add_T_vector!(I, a, v1, v2, q_b, q_c; fdmn=n)
        if q_a ≠ q_b
            add_U_vector!(I, a, v2, q_a, q_b; fdmn=n)
        end
        if q_c ≠ q_d
            add_U_vector!(I, a, v2, q_c, q_d; fdmn=n)
        end
    end
end