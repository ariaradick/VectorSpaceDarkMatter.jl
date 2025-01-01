
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
    PartialRate(K::Vector{A}, model::ModelDMSM, v_basis::RadialBasis,
                q_basis::RadialBasis)

Stores the flattened partial rate matrix along with the relevant model and
basis parameters.
"""
struct PartialRate{A}
    K::Vector{A}
    model::ModelDMSM
    v_basis::RadialBasis
    q_basis::RadialBasis
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

function get_ℓmax(args...; ell_max=nothing)
    ℓmax = minimum([maximum([lm[1] for lm in pf.lm]) for pf in args])
    if !(isnothing(ell_max))
        ℓmax = min(ell_max, ℓmax)
    end
    return ℓmax
end

function _pr(ellmax, model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
    use_measurements=true)

    nv_max = size(pfv.fnlm)[1]-1
    nq_max = size(pfq.fnlm)[1]-1

    mcI = kinematic_I((ellmax, nv_max, nq_max), model, pfv.radial_basis, pfq.radial_basis)
    mcK = collect(Iterators.flatten([get_mcalK_ell(pfv, pfq, ell, mcI[:,:,ell+1]; 
           use_measurements=use_measurements) for ell in 0:ellmax]))
    
    return mcK*pfv.radial_basis.umax^3
end

"""
    partial_rate(model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
                 ell_max=nothing, use_measurements=true)

Calculates the partial rate matrix, stored as a flattened vector, for a given
`model`, velocity distribution (`pfv`), and material form factor (`pfq`).

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `pfv` and `pfq`.
"""
function partial_rate(model::ModelDMSM, pfv::ProjectedF, pfq::ProjectedF; 
    ell_max=nothing, use_measurements=true)
    ℓmax = get_ℓmax(pfv, pfq; ell_max=ell_max)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)
    return PartialRate(mcK, model, pfv.radial_basis, pfq.radial_basis)
end

"""
    writeK(outfile, mcK::PartialRate)

Writes a partial rate matrix ``\\mathcal{K}`` stored in `mcK` to `outfile`.
"""
function writeK(outfile, mcK::PartialRate)
    vtype = typeof(mcK.v_basis)
    vmax = mcK.v_basis.umax
    qtype = typeof(mcK.q_basis)
    qmax = mcK.q_basis.umax
    model = mcK.model

    open(outfile, "w") do io
        write(io, "#, v_type: $vtype, vMax: $vmax, q_type: $qtype, qMax: $qmax\n")
        write(io, "#, fdm_n: $(model.fdm_n), mX: $(model.mX), mSM: $(model.mSM), deltaE: $(model.deltaE)\n")
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
    readK(infile[, v_basis::RadialBasis, q_basis::RadialBasis]; use_err=true)

Reads partial rate matrix from `infile`. Can optionally manually define the 
bases via `v_basis` and `q_basis` arguments, otherwise will attempt to read
from the header of the csv file. Note that automatic basis reading only works
for `Wavelet` bases.

`use_err` : Whether or not to load the uncertainties.
"""
function readK(infile, v_basis::RadialBasis, q_basis::RadialBasis; use_err=true)
    input = readdlm(infile, ','; comments=true)
    if (size(input)[2] == 1) || !(use_err)
        Kvec = @view input[:]
    elseif size(input)[2] == 2
        Kvals = @view input[:,1]
        Kerrs = @view input[:,2]
        Kvec = (Kvals .± Kerrs)
    end

    B = _get_K_info(infile)
    model_params = parse.(Float64, [B["fdm_n"], B["mX"], B["mSM"], B["deltaE"]])

    mcK = PartialRate(Kvec, ModelDMSM(model_params...), v_basis, q_basis)

    return mcK
end

function readK(infile; use_err=true)
    B = _get_K_info(infile)
    if (B["v_type"] in ["Wavelet", "wavelet"]) && (B["q_type"] in ["Wavelet", "wavelet"])
        vbasis = Wavelet(parse(Float64, B["vMax"]))
        qbasis = Wavelet(parse(Float64, B["qMax"]))
    end
    return readK(infile, vbasis, qbasis; use_err=use_err)
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

Using measurements here is currently slow.
"""
function rate(R::T, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false
              ) where T<:Union{Quaternion,Rotor}

    ℓmax = get_ℓmax(pfv, pfq; ell_max=ell_max)
    vmax = pfv.radial_basis.umax
    qmax = pfq.radial_basis.umax
    
    G = G_matrices(R, ℓmax)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)

    return dot(mcK, G)*vmax^2/qmax
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
    vmax = pfv.radial_basis.umax
    qmax = pfq.radial_basis.umax

    D = D_prep(ℓmax)
    G = G_matrices(one(RotorF64), ℓmax)
    mcK = _pr(ℓmax, model, pfv, pfq; use_measurements=use_measurements)

    res = zeros(Float64, size(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = dot(mcK,G)
    end
    return res .* vmax^2 ./ qmax
end

function rate(R, exp::ExposureFactor, model::ModelDMSM, pfv::ProjectedF, 
              pfq::ProjectedF; ell_max=nothing, use_measurements=false)
    r = rate(R,model,pfv,pfq;ell_max=ell_max,use_measurements=use_measurements)
    return @. r*exp.total
end

"""
    rate(R, mcK::PartialRate)

Calculates the rate for a given partial rate matrix `mcK` and rotation `R`
given as a quaternion or rotor (to use other rotations, see `Quaternionic.jl`'s
conversion methods). `R` can also be a vector of quaternions or rotors.

The maximum ``\\ell`` that is used is the minimum of the specified `ell_max` and
the maximum ``\\ell`` for each of `pfv` and `pfq`.

Measurements are not supported here (yet).
"""
function rate(R::T, mcK::PartialRate) where T<:Union{Quaternion,Rotor}
    vmax = K.v_basis.umax
    qmax = K.q_basis.umax

    G = G_matrices(R, ℓmax)
    Kvec = Measurements.value.(mcK.K)
    return dot(Kvec, G)*vmax^2/qmax
end

function rate(R::Array{T}, mcK::PartialRate) where T<:Union{Quaternion,Rotor}
    vmax = K.v_basis.umax
    qmax = K.q_basis.umax

    D = D_prep(ℓmax)
    G = G_matrices(one(RotorF64), ℓmax)

    Kvec = Measurements.value.(mcK.K)

    res = zeros(Float64, size(R))
    for i in eachindex(R)
        D_matrices!(D, R[i])
        G_matrices!(G, D)
        res[i] = dot(Kvec,G)
    end
    return res .* vmax^2 ./ qmax
end

rate(R, exp::ExposureFactor, mcK::PartialRate) = exp.total .* rate(R, mcK)