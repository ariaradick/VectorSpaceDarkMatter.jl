function _get_im_vals(pf::ProjectedF, ell)
    i_vals = findall(x -> x[1] == ell, pf.lm)
    Nj = length(i_vals)
    m_vals = zeros(Int, Nj)
    for j in 1:Nj
        m_vals[j] = pf.lm[i_vals[j]][2]
    end
    return (i_vals, m_vals)
end

"""
    get_mcalK_ell(pfv::ProjectedF, pfq::ProjectedF, ell, I_ell;
        use_measurements=true)

Calculates K^l_{m,m'} = <gχ | n l m> I^l_{n,n'} <n' l m' | f_S^2>
for a given ell. Takes a pre-computed I_ell array.
"""
function get_mcalK_ell(pfv::ProjectedF, pfq::ProjectedF, ell, I_ell;
        use_measurements=true)
    iv_vals, mv_vals = _get_im_vals(pfv, ell)
    iq_vals, mq_vals = _get_im_vals(pfq, ell)
    Nm = 2*ell+1

    fv_vals = Measurements.value.(pfv.fnlm)
    fq_vals = Measurements.value.(pfq.fnlm)
    Il_vals = Measurements.value.(I_ell)

    K_val = zeros(Float64, (Nm, Nm))

    eltypes = eltype.((pfv.fnlm, pfq.fnlm, I_ell))

    if !use_measurements || (sum(eltypes .<: Measurement) == 0)
        for jq in eachindex(mq_vals)
            kq = mq_vals[jq] + ell + 1
            for jv in eachindex(mv_vals)
                kv = mv_vals[jv] + ell + 1
                K_val[kv, kq] = fv_vals[:,iv_vals[jv]]' * Il_vals *
                                fq_vals[:,iq_vals[jq]]
            end
        end
        return K_val
    
    else
        fv_errs = Measurements.uncertainty.(pfv.fnlm)
        fq_errs = Measurements.uncertainty.(pfq.fnlm)
        Il_errs = Measurements.uncertainty.(I_ell)
    
        K_err = zeros(Float64, (Nm, Nm))
        for jq in eachindex(mq_vals)
    
            kq = mq_vals[jq] + ell + 1
            fqn = @view fq_vals[:,iq_vals[jq]]
            fqn_err = @view fq_errs[:,iq_vals[jq]]
    
            for jv in eachindex(mv_vals)
    
                kv = mv_vals[jv] + ell + 1
                fvn = @view fv_vals[:,iv_vals[jv]]
                fvn_err = @view fv_errs[:,iv_vals[jv]]
    
                k_mat_err = @. sqrt( (fvn_err' * fqn * Il_vals)^2 +
                                     (fqn_err * fvn' * Il_vals)^2 +
                                     (Il_errs * fvn' * fqn)^2 )
    
                K_val[kv, kq] = fvn' * Il_vals * fqn
                K_err[kv, kq] = sqrt(dot(k_mat_err, k_mat_err))
            end
        end

        return K_val .± K_err
    end
end

"""
    rate(R::T, model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
        ell_max=nothing, use_measurements=false) where T<:Union{Quaternion,Rotor}

Calculates the rate for a given model, projected g_χ (pfv), projected fs2 (pfq),
and rotation given as a quaternion or rotor (if you want other rotations, see
Quaternionic.jl's conversion methods.)

The maximum ell that is used is the minimum of the specified ell_max and the
maximum ell for each of pfv and pfq.

Using measurements here is currently slow.
"""
function rate(R::T, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false
              ) where T<:Union{Quaternion,Rotor}

    ℓmax = min(maximum([lm[1] for lm in pfv.lm]),
               maximum([lm[1] for lm in pfq.lm]))
    if !(isnothing(ell_max))
        ℓmax = min(ell_max, ℓmax)
    end
        
    nv_max = size(pfv.fnlm)[1]-1
    nq_max = size(pfq.fnlm)[1]-1

    vmax = pfv.radial_basis.umax
    qmax = pfq.radial_basis.umax
    
    G = G_matrices(R, ℓmax)
    mcI = I_lvq_vec((ℓmax, nv_max, nq_max), model, pfv.radial_basis, pfq.radial_basis)
    mcK = collect(Iterators.flatten([get_mcalK_ell(pfv, pfq, ell, mcI[:,:,ell+1]; 
           use_measurements=use_measurements) for ell in 0:ℓmax]))

    return dot(mcK, G)*vmax^5/qmax
end

"If you supply a vector of rotations, will properly (quickly) calculate the
rate for each."
function rate(R::Array{T}, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false
              ) where T<:Union{Quaternion,Rotor}
    ℓmax = min(maximum([lm[1] for lm in pfv.lm]),
               maximum([lm[1] for lm in pfq.lm]))
    if !(isnothing(ell_max))
        ℓmax = min(ell_max, ℓmax)
    end
    nv_max = size(pfv.fnlm)[1]-1
    nq_max = size(pfq.fnlm)[1]-1

    vmax = pfv.radial_basis.umax
    qmax = pfq.radial_basis.umax

    D = D_prep(ℓmax)
    G = G_matrices(one(RotorF64), ℓmax)
    mcI = I_lvq_vec((ℓmax, nv_max, nq_max), model, pfv.radial_basis, pfq.radial_basis)
    mcK = collect(Iterators.flatten([get_mcalK_ell(pfv, pfq, ell, mcI[:,:,ell+1]; 
           use_measurements=use_measurements) for ell in 0:ℓmax]))

    res = zeros(Float64, size(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = dot(mcK,G)
    end
    return res .* vmax^5 ./ qmax
end