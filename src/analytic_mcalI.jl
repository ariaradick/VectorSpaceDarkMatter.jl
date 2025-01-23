function _b_nk_int(n, k, x)
    res = 0.0
    for j in 0:k
        comb = binomial(Float64(k),j)
        ipower = j + (n - k)/2 + 1
        if ipower ≈ 0
            res += comb*log(x)
        else
            res += comb * x^ipower / ipower
        end
    end
    res *= 0.5
    return res
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
    elseif (α > 0) && (α%1 == 0)
        res = (-1)^α * log(1+x)/α + (1+x)^α / (2*α^2)
        for j in 1:(Int(α)-1)
            binm = binomial(α, j) + binomial(α-1, j)
            res += (-1)^(α-j) * (1+x)^j * binm / (2*α*j)
        end
        return res
    elseif (α%1 == 0)
        res = (-1)^α * log((1+x)/x)/α - (1+1/x)^(-α) / (2*α^2)
        for j in 1:(-Int(α)-1)
            binm = binomial(-α, j) + binomial(-α-1, j)
            res += (-1)^(α+j) * ((1+x)/x)^j * binm / (2*α*j)
        end
        return res
    else
        return x^α / α^2 * (0.5 - hyp2f1(1, α, 1+α, -x))
    end
end

function _v_nm_int(n, m, x)
    res = 0.0
    for j in 0:(m+2)
        comb = binomial(m+2,j)
        ipower = j + (n-m)/2
        if ipower ≈ 0
            res += comb*log(x)
        else
            res += comb * x^ipower / ipower
        end
    end
    res *= 0.5
    return res
end

function _s_nm_int(n, m, x)
    res = _v_nm_int(n,m,x) * log(sqrt(x)/(1+x))
    for j in 0:(m+2)
        ipower = j + (n-m)/2
        comb = binomial((m+2),j)
        res += 0.5*comb*_c_alpha_int(ipower, x)
    end
    return res
end

function T_matrix(ℓmax)
    res = zeros(Float64, (ℓmax+1, ℓmax+1))
    for ell in 0:ℓmax
        for k in (ell%2):2:ell
            res[k+1,ell+1] = 2^(ell-k)*gamma(0.5*(k+ell+1)) / (gamma(k+1) *
                             gamma(ell-k+1) * gamma(0.5*(k-ell+1)))
        end
    end
    return LowerTriangular(transpose(res))
end

function add_T_vector!(I, a, n, m, v1, v2, q1, q2)
    for i in eachindex(I)
        if i==(m+3)
            I[i] += a * (_b_nk_int(n, m+2, q2^2) - _b_nk_int(n, m+2, q1^2)) *
                    log(v2/v1)
        else
            k = i-1
            I[i] += a * (_b_nk_int(n, k, q2^2) - _b_nk_int(n, k, q1^2)) * 
                    (v2^(m+2-k) - v1^(m+2-k)) / (m+2-k)
        end
    end
end

function add_U_vector!(I, a, n, m, v2, q1, q2)
    dBm2 = _b_nk_int(n,m+2,q2^2) - _b_nk_int(n,m+2,q1^2)
    for i in eachindex(I)
        if i==(m+3)
            I[i] += a*(log(2*v2)*dBm2 + _s_nm_int(n,m,q2^2) - 
                    _s_nm_int(n,m,q1^2))
        else
            k = i-1
            I[i] += a*(v2^(m+2-k)  * (_b_nk_int(n,k,q2^2) - 
                    _b_nk_int(n,k,q1^2)) - 2.0^(k-m-2)*dBm2) / (m+2-k)
        end
    end
end

function not_mI_star!(I, a, n, m, v1, v2, q1, q2)
    if v1==v2 || q1==q2
        return # No integration volume
    end
    if q2 < q1
        error("q12 must be ordered.")
    end
    if v2 < v1
        error("v12 must be ordered.")
    end
    include_R2 = true
    if v2 < 1.0
        # v2 is below the velocity threshold. mcalI=0
        return
    end
    tilq_m,tilq_p = v2 - sqrt(v2^2-1.0), v2 + sqrt(v2^2-1.0)
    if tilq_m > q2 || tilq_p < q1
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return
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
        add_U_vector!(I, a, n, m, v2, q_A, q_B)
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

        add_T_vector!(I, a, n, m, v1, v2, q_b, q_c)
        if q_a ≠ q_b
            add_U_vector!(I, a, n, m, v2, q_a, q_b)
        end
        if q_c ≠ q_d
            add_U_vector!(I, a, n, m, v2, q_c, q_d)
        end
    end
end