
"""
    ExposureFactor(N_T, sigma0, rhoX)

Necessary constants for calculating ``k_0``, the overall prefactor for the rate.

`N_T` : number of individual targets, ``N_T = M_T / m_{\\textrm{cell}}`` 
where ``M_T`` is the mass of the target, and ``m_{\\textrm{cell}}`` is the mass
of the unit cell.

`sigma0` : Reference cross-section in cm^2, ``\\bar{\\sigma}_0``

`rhoX` : local DM density in GeV/cm^3, ``\\rho_\\chi``
"""
struct ExposureFactor
    N_T::Float64
    sigma0::Float64
    rhoX::Float64
    total::Float64

    function ExposureFactor(N_T, sigma0, rhoX)
        tot = N_T*ccms*1e9*sigma0*rhoX
        return new(N_T, sigma0, rhoX, tot)
    end
end

"""
    McalK(K::Vector{A}, ell_max::Int64, vmax::Float64, qmax::Float64)

Stores the flattened ``\\mathcal{K}`` matrix along with relevant parameters.
"""
struct McalK{A<:Union{Measurement,Float64}}
    K::Vector{A}
    ell_max::Int64
    vmax::Float64
    qmax::Float64
end

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

    if eltype(pfv.fnlm) <: Measurement
        fv_vals = Measurements.value.(pfv.fnlm)
    else
        fv_vals = pfv.fnlm
    end
    if eltype(pfq.fnlm) <: Measurement
        fq_vals = Measurements.value.(pfq.fnlm)
    else
        fq_vals = pfq.fnlm
    end
    Il_vals = I_ell

    K_val = zeros(Float64, (Nm, Nm))

    eltypes = eltype.((pfv.fnlm, pfq.fnlm, I_ell))

    if !use_measurements || (sum(eltypes .<: Measurement) == 0)
        for jq in eachindex(mq_vals)
            kq = mq_vals[jq] + ell + 1
            fqj = @view fq_vals[:,iq_vals[jq]]
            for jv in eachindex(mv_vals)
                fvj = @view fv_vals[:,iv_vals[jv]]
                kv = mv_vals[jv] + ell + 1
                K_val[kv, kq] = dot(fvj, Il_vals, fqj)
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

function get_ℓmax(args...; ell_max=nothing)
    ℓmax = minimum([maximum([lm[1] for lm in pf.lm]) for pf in args])
    if !(isnothing(ell_max))
        ℓmax = min(ell_max, ℓmax)
    end
    return ℓmax
end

function _mcalK_ell!(K_ell, I_ell, ell, fv, iv, mv, fq, iq, mq)
    for jq in eachindex(mq)
        kq = mq[jq] + ell + 1
        fqj = @view fq[:,iq[jq]]
        for jv in eachindex(mv)
            fvj = @view fv[:,iv[jv]]
            kv = mv[jv] + ell + 1
            K_ell[kv, kq] = dot(fvj, I_ell, fqj)
        end
    end
end

function _mcalK_ell_err!(K_ell, K_ell_err, I_ell, ell, fv, fv_errs, iv, mv, 
                         fq, fq_errs, iq, mq)
    for jq in eachindex(mq)
    kq = mq[jq] + ell + 1
    fqn = @view fq[:,iq[jq]]
    fqn_err = @view fq_errs[:,iq[jq]]

        for jv in eachindex(mv)
            kv = mv[jv] + ell + 1
            fvn = @view fv[:,iv[jv]]
            fvn_err = @view fv_errs[:,iv[jv]]

            k_mat_err = @. sqrt( (fvn_err' * fqn * I_ell)^2 +
                                    (fqn_err * fvn' * I_ell)^2 )

            K_ell[kv, kq] = dot(fvn, I_ell, fqn)
            K_ell_err[kv, kq] = sqrt(dot(k_mat_err, k_mat_err))
        end
    end
end

function _get_pf_vals(pf::ProjectedF{A,B}) where {A<:Measurement, B<:RadialBasis}
    return Measurements.value.(pf.fnlm)
end

function _get_pf_vals(pf::ProjectedF{Float64,B}) where B<:RadialBasis
    return pf.fnlm
end

function _pr(ellmax, pfv::ProjectedF, pfq::ProjectedF, mcI; 
             use_measurements=true)
    vmax = pfv.radial_basis.umax

    fv_vals = _get_pf_vals(pfv)
    fq_vals = _get_pf_vals(pfq)

    K_vals = zeros(Float64, WignerDsize(ellmax))

    eltypes = eltype.((pfv.fnlm, pfq.fnlm, mcI))

    if !use_measurements || (sum(eltypes .<: Measurement) == 0)
        for ell in 0:ellmax
            iv_vals, mv_vals = _get_im_vals(pfv, ell)
            iq_vals, mq_vals = _get_im_vals(pfq, ell)

            I_ell = @view mcI[:,:,ell+1]

            i1 = WignerDindex(ell, -ell, -ell)
            i2 = WignerDindex(ell, ell, ell)
            ksize = 2*ell+1

            K_ell = @views reshape(K_vals[i1:i2], (ksize,ksize))

            _mcalK_ell!(K_ell, I_ell, ell, fv_vals, iv_vals, mv_vals, 
                        fq_vals, iq_vals, mq_vals)
        end
        @. K_vals *= vmax^3
        return K_vals
    else
        K_errs = zeros(Float64, WignerDsize(ellmax))
        fv_errs = Measurements.uncertainty.(pfv.fnlm)
        fq_errs = Measurements.uncertainty.(pfq.fnlm)

        for ell in 0:ellmax
            iv_vals, mv_vals = _get_im_vals(pfv, ell)
            iq_vals, mq_vals = _get_im_vals(pfq, ell)

            I_ell = @view mcI[:,:,ell+1]

            i1 = WignerDindex(ell, -ell, -ell)
            i2 = WignerDindex(ell, ell, ell)
            ksize = 2*ell+1

            K_ell = @views reshape(K_vals[i1:i2], (ksize,ksize))
            K_err = @views reshape(K_errs[i1:i2], (ksize,ksize))

            _mcalK_ell_err!(K_ell, K_err, I_ell, ell, fv_vals, fv_errs, iv_vals,
                            mv_vals, fq_vals, fq_errs, iq_vals, mq_vals)
        end
        @. K_vals *= vmax^3
        @. K_errs *= vmax^3
        return K_vals .± K_errs
    end
end

function _pr(ellmax, model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
    use_measurements=true)
    nv_max = size(pfv.fnlm)[1]-1
    nq_max = size(pfq.fnlm)[1]-1
    mcI = kinematic_I((ellmax, nv_max, nq_max), model, pfv.radial_basis, 
                       pfq.radial_basis)
    return _pr(ellmax, pfv, pfq, mcI; use_measurements=use_measurements)
end

"""
    get_mcalK(model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
                 ell_max=nothing, use_measurements=true)

Calculates the matrix ``\\mathcal{K}``, stored as a flattened vector, for a 
given `model`, velocity distribution (`pfv`), and material form factor (`pfq`).

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `pfv` and `pfq`.
"""
function get_mcalK(model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
    ell_max=nothing, use_measurements=true)
    ℓmax = get_ℓmax(pfv, pfq; ell_max=ell_max)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)
    return McalK(mcK, ℓmax, pfv.radial_basis.umax, pfq.radial_basis.umax)
end

"""
    get_mcalK(mcI, pfv::ProjectedF, pfq::ProjectedF; 
              ell_max=nothing, use_measurements=true)

Calculates the matrix ``\\mathcal{K}``, stored as a flattened vector, for a 
given kinematic scattering matrix `mcI`, velocity distribution (`pfv`), and 
material form factor (`pfq`).

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `mcI`, `pfv`, `pfq`.
"""
function get_mcalK(mcI, pfv::ProjectedF, pfq::ProjectedF; 
    ell_max=nothing, use_measurements=true)
    
    ell_I = size(mcI)[3]
    lmax = min(ell_max,ell_I)

    ℓmax = get_ℓmax(pfv, pfq; ell_max=lmax)
    mcK = _pr(ℓmax, pfv, pfq, mcI; use_measurements=use_measurements)
    return McalK(mcK, ℓmax, pfv.radial_basis.umax, pfq.radial_basis.umax)
end

"""
    partial_rate(mcK::McalK)

Returns the partial rate matrix ``K = v_{\\textrm{max}}^2 / q_{\\textrm{max}}^2
\\mathcal{K}``
"""
function partial_rate(mcK::McalK)
    return mcK.K .* (mcK.vmax^2 / mcK.qmax)
end

"""
    writeK(outfile, mcK::McalK)

Writes a partial rate matrix ``\\mathcal{K}`` stored in `mcK` to `outfile`.
"""
function writeK(outfile, mcK::McalK)
    ellmax = mcK.ell_max
    vmax = mcK.vmax
    qmax = mcK.qmax

    open(outfile, "w") do io
        write(io, "#, ell_max: $ellmax, vMax: $vmax, qMax: $qmax\n")
        if eltype(mcK.K) == Float64
            write(io, "#, K\n")
            writedlm(io, mcK.K, ',')
        elseif eltype(mcK.K) <: Measurement
            write(io, "#, K.val, K.err\n")
            Kvals = Measurements.value.(mcK.K)
            Kerrs = Measurements.uncertainty.(mcK.K)
            writedlm(io, hcat(Kvals, Kerrs), ',')
        end
    end

    return
end

function _get_K_info(infile)
    B = Dict{String,String}()
    open(infile) do io
        for l in eachline(io)
            if l[1] == '#'
                stuff = split.(split(filter(x -> !isspace(x), l[2:end]), ','), ':')
                filter!(x -> length(x) == 2, stuff)
                for s in stuff
                    B[s[1]] = s[2]
                end
            else
                break
            end
        end
    end
    return B
end

"""
    readK(infile[, vmax, qmax]; use_err=true)

Reads partial rate matrix from `infile`. Can optionally manually define the 
basis maximums via `vmax` and `qmax` arguments, otherwise will attempt to read
from the header of the csv file.

`use_err` : Whether or not to load the uncertainties.
"""
function readK(infile, vmax, qmax; use_err=true)
    input = readdlm(infile, ','; comments=true)
    if (size(input)[2] == 1) || !(use_err)
        Kvec = input[:]
    elseif size(input)[2] == 2
        Kvals = input[:,1]
        Kerrs = input[:,2]
        Kvec = (Kvals .± Kerrs)
    end

    B = _get_K_info(infile)
    ellmax = parse(Int64, B["ell_max"])

    mcK = McalK(Kvec, ellmax, vmax, qmax)
    return mcK
end

function readK(infile; use_err=true)
    B = _get_K_info(infile)
    vmax = parse(Float64, B["vMax"])
    qmax = parse(Float64, B["qMax"])
    return readK(infile, vmax, qmax; use_err=use_err)
end

"""
    rate(R, model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
        ell_max=nothing, use_measurements=false)

Calculates the rate for a given model, projected ``g_\\chi`` (`pfv`), projected 
``f_s^2`` (`pfq`), and rotation `R` given as a quaternion or rotor (if you want 
other rotations, see `Quaternionic.jl`'s conversion methods). `R` can also
be a vector of quaternions or rotors.

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `pfv` and `pfq`.

Using measurements here is currently slow and not recommended.
"""
function rate(R::T, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false
              ) where T<:Union{Quaternion,Rotor}

    ℓmax = get_ℓmax(pfv, pfq; ell_max=ell_max)
    
    G = G_matrices(R, ℓmax)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)

    return dot(mcK, G)
end

function rate(model::ModelDMSM, pfv::ProjectedF, 
    pfq::ProjectedF; ell_max=nothing, use_measurements=false)
    
    rate(one(RotorF64), model, pfv, pfq; ell_max=ell_max,
         use_measurements=use_measurements)
end

function rate(R::Array{T}, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false
              ) where T<:Union{Quaternion,Rotor}

    ℓmax = get_ℓmax(pfv, pfq; ell_max=ell_max)

    D = D_prep(ℓmax)
    G = G_matrices(one(RotorF64), ℓmax)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)

    res = zeros(Float64, size(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = dot(mcK,G)
    end
    return res
end

function rate(R, exp::ExposureFactor, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false,
              t_unit='s')
    if t_unit == 's'
        pref = 1.0
    elseif t_unit == 'y'
        pref = SECONDS_PER_YEAR
    end
    r = rate(R,model,pfv,pfq;ell_max=ell_max,use_measurements=use_measurements)
    @. r *= pref*exp.total*vmax^2/(qmax)
    return r
end

"""
    rate(R, mcK::McalK)

Calculates the rate for a given partial rate matrix `mcK` and rotation `R`
given as a quaternion or rotor (to use other rotations, see `Quaternionic.jl`'s
conversion methods). `R` can also be a vector of quaternions or rotors.

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `pfv` and `pfq`.

Measurements are not supported here (yet).
"""
function rate(R::T, mcK::McalK) where T<:Union{Quaternion,Rotor}
    G = G_matrices(R, mcK.ell_max)
    Kvec = Measurements.value.(mcK.K)
    return dot(Kvec, G)
end

function rate(R::Array{T}, mcK::McalK) where T<:Union{Quaternion,Rotor}
    D = D_prep(mcK.ell_max)
    G = G_matrices(one(RotorF64), mcK.ell_max)

    Kvec = Measurements.value.(mcK.K)

    res = zeros(Float64, size(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = dot(Kvec,G)
    end
    return res
end

function rate(R, exp::ExposureFactor, mcK::McalK; t_unit='s')
    if t_unit == 's'
        pref = 1.0
    elseif t_unit == 'y'
        pref = SECONDS_PER_YEAR
    end
    vmax = mcK.v_basis.umax
    qmax = mcK.q_basis.umax
    @. pref * exp.total * rate(R, mcK) * vmax^2 / (qmax)
end

"""
    N_events(R, T_exp_s, exp::ExposureFactor, model::ModelDMSM, 
    pfv::ProjectedF, pfq::ProjectedF; ell_max=nothing, use_measurements=false)

Returns the total number of expected events for a given rotation `R` and 
exposure time `T_exp_s` in seconds.
"""
function N_events(R, T_exp_s, exp::ExposureFactor, model::ModelDMSM, 
    pfv::ProjectedF, pfq::ProjectedF; ell_max=nothing, use_measurements=false)
    rate(R, exp, model, pfv, pfq; ell_max=ell_max, 
    use_measurements=use_measurements) * T_exp_s
end

"""
    N_events(R, T_exp_s, exp::ExposureFactor, mcK::McalK)

Returns the total number of expected events for a given rotation `R` and 
exposure time `T_exp_s` in seconds.
"""
function N_events(R, T_exp_s, exp::ExposureFactor, mcK::McalK)
    rate(R,exp,mcK) * T_exp_s
end