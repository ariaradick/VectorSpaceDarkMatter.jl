"""
    ModelDMSM(beta, gamma, mX, mSM, deltaE)

Stores the relevant model parameters.

`beta` : Two times the power of ``q`` in the dark matter form factor, 
         ``F_{DM} \\sim (q / \\alpha m_e)^{\\beta/2}``

`gamma` : Two times the power of ``v`` in the dark matter form factor, 
         ``F_{DM} \\sim v^{\\gamma/2}``

`mX` : dark matter mass in eV

`mSM` : mass of target particle in eV

`deltaE` : discrete final state energy in eV
"""
struct ModelDMSM
    beta::Int
    gamma::Int
    mX::Float64
    mSM::Float64
    deltaE::Float64
end

"""
    ModelDMSM(fdm_n, mX, mSM, deltaE)

Stores the relevant model parameters. Can be called with just `fdm_n` to
correspond with typically-used dark matter form factors.

`fdm_n` : The power ``n`` of ``1/q`` in the dark matter form factor, 
         ``F_{DM} \\sim (\\alpha m_e / q)^{n}``

`mX` : dark matter mass in eV

`mSM` : mass of target particle in eV

`deltaE` : discrete final state energy in eV
"""
function ModelDMSM(fdm_n, mX, mSM, deltaE)
    ModelDMSM(-2*fdm_n, 0, mX, mSM, deltaE)
end

function _getABval(n, basis::Wavelet)
    haar_sph_value(n)
end

function _getABval(n, basis::Tophat)
    x_n, x_np1 = basis.xi[n+1], basis.xi[n+2]
    return tophat_value(x_n, x_np1)
end

function _fill_I!(Imat, tmat, lnvnq_max, model::ModelDMSM, 
                  v_basis::RadialBasis, q_basis::RadialBasis)
    nv_max, nq_max = lnvnq_max[2:3]
    mX = model.mX
    n = model.beta
    m = model.gamma
    DeltaE = model.deltaE

    v0 = v_basis.umax
    q0 = q_basis.umax

    qStar = sqrt(2*mX*DeltaE)
    vStar = qStar/mX

    for nq in 0:nq_max
        q_base = _base_of_support_n(nq, q_basis).*q0
        AB_q = _getABval(nq, q_basis)

        for nv in 0:nv_max
            v_base = _base_of_support_n(nv, v_basis).*v0
            AB_v = _getABval(nv, v_basis)

            temp = @view Imat[:, nv+1, nq+1]
            for i in eachindex(AB_v)
                for j in eachindex(AB_q)
                    v1_star = v_base[i] / vStar
                    v2_star = v_base[i+1] / vStar
                    q1_star = q_base[j] / qStar
                    q2_star = q_base[j+1] / qStar
                    not_mI_star!(temp, AB_v[i]*AB_q[j], n, m, v1_star, v2_star,
                                 q1_star, q2_star)
                end
            end

            lmul!(tmat, temp)
        end
    end
end

function kinematic_I(lnvnq_max, model::ModelDMSM, 
    v_basis::RadialBasis, q_basis::RadialBasis)
    l_max, nv_max, nq_max = lnvnq_max

    mX = model.mX
    n = model.beta
    m = model.gamma
    DeltaE = model.deltaE
    mSM = model.mSM

    v0 = v_basis.umax
    q0 = q_basis.umax

    muSM = mX*mSM/(mX+mSM)

    qStar = sqrt(2*mX*DeltaE)
    vStar = qStar/mX

    commonFactor = ((q0/v0)^3/(2*mX*muSM^2) * (2*DeltaE/(q0*v0))^2 *
                    (qStar/q0_fdm)^n * vStar^m)

    tmatrix = T_matrix(l_max)
    res = zeros(Float64, (l_max+1, nv_max+1, nq_max+1))
    _fill_I!(res, tmatrix, lnvnq_max, model, v_basis::RadialBasis, 
             q_basis::RadialBasis)
    @. res *= commonFactor

    return permutedims(res,[2,3,1])
end