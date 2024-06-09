
function getI_lvq_analytic(lnvnq, mX, fdm_n, DeltaE, muSM, 
        v_basis::Wavelet, q_basis::Wavelet)
    ell, nv, nq = lnvnq
    v0 = v_basis.umax
    q0 = q_basis.umax
    qStar = sqrt(2*mX*DeltaE)
    vStar = qStar/mX

    commonFactor = ((q0/v0)^3/(2*mX*muSM^2) * (2*DeltaE/(q0*v0))^2
                        *(q0_fdm/qStar)^(2*fdm_n))
    
    v_base = _base_of_support_n(nv, v_basis).*v0
    AB_v = haar_sph_value(nv)
    q_base = _base_of_support_n(nq, q_basis).*q0
    AB_q = haar_sph_value(nq)

    res = 0.0
    for i in eachindex(AB_v)
        for j in eachindex(AB_q)
            v12_star = [v_base[i], v_base[i+1]] ./ vStar
            q12_star = [q_base[j], q_base[j+1]] ./ qStar
            res += AB_v[i]*AB_q[j]*mI_star(ell, fdm_n, v12_star, q12_star)
        end
    end
    return commonFactor*res
end