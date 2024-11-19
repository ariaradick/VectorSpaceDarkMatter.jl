"""
    ModelDMSM(fdm_n, mX, mSM, deltaE)

Stores the relevant model parameters.

`fdm_n` : The power of the dark matter form factor ``(\\alpha m_e / q)^{\\texttt{fdm_n}}``

`mX` : dark matter mass in eV

`mSM` : mass of target particle in eV

`deltaE` : discrete final state energy in eV
"""
struct ModelDMSM
    fdm_n::Int
    mX::Float64
    mSM::Float64
    deltaE::Float64
end

function _getABval(n, basis::Wavelet)
    haar_sph_value(n)
end

function _getABval(n, basis::Tophat)
    x_n, x_np1 = basis.xi[n+1], basis.xi[n+2]
    return tophat_value(x_n, x_np1)
end

function getI_lvq_analytic(lnvnq, model::ModelDMSM, v_basis::RadialBasis, 
        q_basis::RadialBasis)
    mX = model.mX
    fdm_n = model.fdm_n
    DeltaE = model.deltaE
    mSM = model.mSM

    ell, nv, nq = lnvnq
    v0 = v_basis.umax
    q0 = q_basis.umax

    muSM = mX*mSM/(mX+mSM)

    qStar = sqrt(2*mX*DeltaE)
    vStar = qStar/mX

    commonFactor = ((q0/v0)^3/(2*mX*muSM^2) * (2*DeltaE/(q0*v0))^2
                        *(q0_fdm/qStar)^(2*fdm_n))
    
    v_base = _base_of_support_n(nv, v_basis).*v0
    AB_v = _getABval(nv, v_basis)
    q_base = _base_of_support_n(nq, q_basis).*q0
    AB_q = _getABval(nq, q_basis)

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

function I_lvq(lnvnq_max, model::ModelDMSM, v_basis::RadialBasis, 
        q_basis::RadialBasis)
    l_max, nv_max, nq_max = lnvnq_max

    res = zeros(Float64, (nv_max+1, nq_max+1, l_max+1))

    Threads.@threads for ell in 0:l_max
        Threads.@threads for nq in 0:nq_max
            Threads.@threads for nv in 0:nv_max
                res[nv+1, nq+1, ell+1] = getI_lvq_analytic((ell, nv, nq),
                                         model, v_basis, q_basis)
            end
        end
    end

    return res
end

function I_lvq_vec(lnvnq_max, model::ModelDMSM, 
    v_basis::RadialBasis, q_basis::RadialBasis)
    l_max, nv_max, nq_max = lnvnq_max

    mX = model.mX
    fdm_n = model.fdm_n
    DeltaE = model.deltaE
    mSM = model.mSM

    v0 = v_basis.umax
    q0 = q_basis.umax

    muSM = mX*mSM/(mX+mSM)

    qStar = sqrt(2*mX*DeltaE)
    vStar = qStar/mX

    commonFactor = ((q0/v0)^3/(2*mX*muSM^2) * (2*DeltaE/(q0*v0))^2
            *(q0_fdm/qStar)^(2*fdm_n))

    tmatrix = T_matrix(l_max)

    res = zeros(Float64, (nv_max+1, nq_max+1, l_max+1))

    @Threads.threads for nq in 0:nq_max
        q_base = _base_of_support_n(nq, q_basis).*q0
        AB_q = _getABval(nq, q_basis)

        @Threads.threads for nv in 0:nv_max
            v_base = _base_of_support_n(nv, v_basis).*v0
            AB_v = _getABval(nv, v_basis)

    # temp = @view res[nv+1, nq+1, :]
            temp = zeros(Float64, l_max+1)
            for i in eachindex(AB_v)
                for j in eachindex(AB_q)
                    v12_star = [v_base[i], v_base[i+1]] ./ vStar
                    q12_star = [q_base[j], q_base[j+1]] ./ qStar
                    # temp .+= AB_v[i]*AB_q[j].*not_mI_star(l_max, fdm_n, v12_star, q12_star)
                    not_mI_star!(temp, AB_v[i]*AB_q[j], fdm_n, v12_star, q12_star)
                end
            end

            res[nv+1, nq+1, :] = (tmatrix * temp)
        end
    end
    return commonFactor .* res
end