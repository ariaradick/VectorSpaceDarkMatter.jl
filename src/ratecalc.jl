function _get_im_vals(pf::ProjectedF, ell)
    i_vals = findall(x -> x[1] == ell, pf.lm)
    Nj = length(i_vals)
    m_vals = zeros(Int, Nj)
    for j in 1:Nj
        m_vals[j] = pf.lm[i_vals[j]][2]
    end
    return (i_vals, m_vals)
end

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

function rate(R::Quaternion, model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF;
              ell_max=nothing)

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
    mcI = I_lvq((ℓmax, nv_max, nq_max), model, pfv.radial_basis, pfq.radial_basis)
    mcK = [get_mcalK_ell(pfv, pfq, ell, mcI[:,:,ell+1]; 
           use_measurements=false) for ell in 0:ℓmax]
    
    return sum(tr.( [ dot(mcK[ell+1], g) for (ell,g) in 
           zip(0:ℓmax, D_iterator(G, ℓmax)) ] )) * vmax^2 / qmax
end

function rate(R::Vector{QuaternionF64}, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing)
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
    G = G_matrices(quaternion(1.0), ℓmax)
    mcI = I_lvq((ℓmax, nv_max, nq_max), model, pfv.radial_basis, pfq.radial_basis)
    mcK = [get_mcalK_ell(pfv, pfq, ell, mcI[:,:,ell+1]; 
           use_measurements=false) for ell in 0:ℓmax]

    res = zeros(Float64, length(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = sum(tr.( [ dot(mcK[ell+1], g) for (ell,g) in 
                 zip(0:ℓmax, D_iterator(G, ℓmax)) ] ))
    end
    return res .* vmax^2 ./ qmax
end