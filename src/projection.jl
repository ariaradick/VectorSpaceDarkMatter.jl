"""
    ProjectedF{A, B}(fnlm::Matrix{A}, radial_basis::B)

Stores the <f | nlm> coefficients and the radial basis that was used to
calculate them. It is assumed that spherical harmonics were used for the
angular parts. `A` is the element type of the matrix that stores `fnlm`
coefficients (should be either `Float64` or `Measurement`). `B` is the type 
of radial basis. `lm` stores a vector of `(l,m)` for the `fnlm` matrix:
`fnlm` is indexed as `[n+1,i]` corresponding to `(l,m) = lm[i]`
"""
struct ProjectedF{A, B<:RadialBasis}
    fnlm::Matrix{A}
    lm::Vector{Tuple{Int64,Int64}}
    radial_basis::B
end

"""
    FCoeffs{A, B}(fnlm::Dict{Tuple{Int64,Int64,Int64}, A},
        radial_basis::B)

Stores the <f | nlm> coefficients as a dictionary with (n,l,m) => f_nlm
and the radial basis that was used to calculate them. It is assumed
that spherical harmonics were used for the angular parts. `A` is 
the element type of the dict that stores `fnlm` coefficients (should 
be either `Float64` or `Measurement`). `B` is the type of radial basis.
"""
struct FCoeffs{A, B<:RadialBasis}
    fnlm::Dict{Tuple{Int64,Int64,Int64}, A}
    radial_basis::B
end

function FCoeffs(rb::RadialBasis; use_measurements=false)
    if use_measurements
        dtype = Measurement{Float64}
    else
        dtype = Float64
    end
    return FCoeffs(Dict{Tuple{Int64,Int64,Int64}, dtype}(), rb)
end

function ProjectedF(fcoeffs::FCoeffs)
    n_max = maximum([k[1] for k in keys(fcoeffs.fnlm)])
    lm_vals = sort( unique([(k[2], k[3]) for k in keys(fcoeffs.fnlm)]) )
    lm_dict = Dict( lm_vals[i] => i for i in eachindex(lm_vals) )
    N_lm = length(lm_vals)

    res = zeros(valtype(fcoeffs.fnlm), (n_max+1, N_lm))

    for (k,v) in fcoeffs.fnlm
        n,l,m = k
        lm_idx = lm_dict[(l,m)]
        res[n+1, lm_idx] = v
    end

    return ProjectedF(res, lm_vals, fcoeffs.radial_basis)
end

function FCoeffs(pf::ProjectedF)
    n_vals = 0:(length(pf.fnlm[:,1])-1)
    nlm = [(n,lm...) for lm in pf.lm for n in n_vals]

    res = Dict( nlm[i] => pf.fnlm[i] for i in eachindex(nlm) )
    return FCoeffs(res, pf.radial_basis)
end

"""
    ProjectF(f, nl_max::Tuple{Int,Int}, radial_basis::RadialBasis; 
             dict=false, use_measurements=false, integ_method=:cubature,
             integ_params=(;))

Evaluates <f | nlm> at each (n,l,m) up to nl_max = (n_max, l_max) and returns
the result as a `ProjectedF`.

`f` : Can be a `Function`, `f_uSph`, `GaussianF`, or `Vector{GaussianF}`. 
      `f_uSph` is preferred if your function has any symmetries, and is not 
      gaussian, as specifying those will greatly speed up evaluation.

`radial_basis` : Either a `Wavelet` or `Tophat`

`dict` : If true, returns an `FCoeffs` instead of a `ProjectedF`, which stores
    the coefficients as a dictionary instead.

`use_measurements` : If `true`, will give the results as a measurement with
    uncertainty given by the integration error.

`integ_method` : Either `:cubature`, `:vegas`, or `:vegasmc`

`integ_params` : keyword arguments to pass to the integrator. If `:cubature`, 
    these are kwargs for `hcubature`. If `:vegas` or `:vegasmc`, these are
    kwargs for `MCIntegration`'s `integrate` method.
"""
function ProjectF(f, nl_max::Tuple{Int,Int}, radial_basis::RadialBasis; 
                    dict=false, use_measurements=false, integ_method=:cubature,
                    integ_params=(;))
    fuSph = f_uSph(f)
    ProjectF(fuSph, nl_max, radial_basis; dict=dict,
    use_measurements=use_measurements, integ_method=integ_method,
    integ_params=integ_params)
end

function ProjectF(f::f_uSph, nl_max::Tuple{Int,Int}, 
        radial_basis::RadialBasis; dict=false, use_measurements=false, 
        integ_method=:cubature, integ_params=(;))

    n_max, l_max = nl_max
    n_vals = 0:n_max
    lm_vals = LM_vals(f, l_max)
    N_lm = length(lm_vals)

    res = zeros(Measurement, (n_max+1, N_lm))

    if integ_method == :cubature
        if !(:atol in keys(integ_params))
            f000 = getFnlm(f, (0,0,0), radial_basis; integ_params=integ_params)
            if :rtol in keys(integ_params)
                int_pars = (integ_params..., atol=f000*integ_params.rtol)
            else
                int_pars = (integ_params..., atol=f000*1e-6)
            end
        end
    else
        int_pars = integ_params
    end

    Threads.@threads for i in 1:N_lm
        ell, m = lm_vals[i]
        for n in n_vals
            res[n+1, i] = getFnlm(f, (n, ell, m), radial_basis;
                            integ_method=integ_method,
                            integ_params=int_pars)
        end
    end

    if use_measurements
        pf = ProjectedF(res, lm_vals, radial_basis)
    else
        pf = ProjectedF(Measurements.value.(res), lm_vals, radial_basis)
    end

    if dict
        return FCoeffs(pf)
    else
        return pf
    end
end

function ProjectF(g::GaussianF, nl_max::Tuple{Int,Int}, 
        radial_basis::RadialBasis; dict=false, use_measurements=false, 
        integ_params=(;))

    u_i, θ_i, φ_i = g.uSph
    n_max, l_max = nl_max
    lm_vals = LM_vals(l_max)
    N_lm = length(lm_vals)

    gnl = zeros(Measurement, (n_max+1, l_max+1))
    res = zeros(Measurement, (n_max+1, N_lm))

    Threads.@threads for ell in 0:l_max
        for n in 0:n_max
            gnl[n+1, ell+1] = getGnl(g, n, ell, radial_basis, 
                              integ_params=integ_params)
        end
    end

    Threads.@threads for i in 1:N_lm
        ell, m = lm_vals[i]
        for n in 0:n_max
            res[n+1,i] = g.c * ylm_real(ell, m, θ_i, φ_i) * gnl[n+1,ell+1] /
                        radial_basis.umax^3
        end
    end

    if use_measurements
        pf = ProjectedF(res, lm_vals, radial_basis)
    else
        pf = ProjectedF(Measurements.value.(res), lm_vals, radial_basis)
    end

    if dict
        return FCoeffs(pf)
    else
        return pf
    end
end

function ProjectF(g::Vector{GaussianF}, nl_max::Tuple{Int,Int}, 
        radial_basis::RadialBasis; dict=false, use_measurements=false,
        integ_params=(;))
    pf = Vector{ProjectedF}()
    lm_vals = LM_vals(nl_max[2])

    for gg in g
        push!(pf, ProjectF(gg, nl_max, radial_basis; 
              use_measurements=use_measurements, integ_params=integ_params))
    end

    if use_measurements
        fnlm_vals = sum([Measurements.value.(p.fnlm) for p in pf])
        fnlm_errs = sqrt.(sum([Measurements.uncertainty.(p.fnlm).^2 for p in pf]))
        fnlm = (fnlm_vals .± fnlm_errs)
        pf = ProjectedF(fnlm, lm_vals, radial_basis)

    else
        fnlm = sum([p.fnlm for p in pf])
        pf = ProjectedF(fnlm, lm_vals, radial_basis)
    end

    if dict
        return FCoeffs(pf)
    else
        return pf
    end
end

"""
    update!(fc::FCoeffs, f, nlm::Tuple{Int,Int,Int}; kwargs...)

Evaluates a particular (n,l,m) coefficient for the function f and the radial
basis fc.radial_basis, and stores the result in fc.fnlm[(n,l,m)]. Will
overwrite existing data. kwargs correspond to the kwargs for `getFnlm`
"""
function update!(fc::FCoeffs, f, nlm::Tuple{Int,Int,Int}; kwargs...)
    fnlm = getFnlm(f, nlm, fc.radial_basis; kwargs...)
    if valtype(fc.fnlm) <: Measurement
        fc.fnlm[nlm] = fnlm
    elseif valtype(fc.fnlm) == Float64
        fc.fnlm[nlm] = fnlm.val
    end
end

function update!(fc::FCoeffs, g::Vector{GaussianF}, nlm::Tuple{Int,Int,Int}; 
        kwargs...)
    fnlms = zeros(Measurement{Float64}, length(g))
    for i in eachindex(g)
        fnlms[i] = getFnlm(g[i], nlm, fc.radial_basis; kwargs...)
    end
    if valtype(fc.fnlm) <: Measurement
        fnlm_val = sum(Measurements.value.(fnlms))
        fnlm_err = sqrt(sum(Measurements.uncertainty.(fnlms).^2))
        fc.fnlm[nlm] = (fnlm_val ± fnlm_err)
    elseif valtype(fc.fnlm) == Float64
        fc.fnlm[nlm] = sum(Measurements.value.(fnlms))
    end
end

"If called with a vector of nlm tuples, runs update! for each (n,l,m)"
function update!(fc::FCoeffs, f, nlm::Vector{Tuple{Int,Int,Int}}; kwargs...)
    update!.((fc,), (f,), nlm; kwargs...)
end

function ProjectF(f, nlm_list::Vector{Tuple{Int,Int,Int}},
    radial_basis::RadialBasis; use_measurements=false, kwargs...)
    fc = FCoeffs(radial_basis; use_measurements=use_measurements)

    update!(fc, f, nlm_list; kwargs...)
    return fc
end

function _get_nvals_wavelet(n_star)
    λ = hindex_LM(n_star)[1]
    nvals = zeros(Int, λ+2)
    nvals[1] = n_star
    for i in 1:λ
        nvals[i+1] = floor(nvals[i]/2)
    end
    return nvals
end

function _get_nvals(x, pf::ProjectedF{A, Wavelet}) where A<:Union{Float64, Measurement}
    n_max = size(pf.fnlm)[1]-1
    λ_max = hindex_LM(n_max)[1]
    μ_star = [floor(Int, x*(2^λ_max))]
    if (x*(2^λ_max))%1 ≈ 0.0
        if x ≈ 1.0
            μ_star[1] -= 1
        else
            push!(μ_star, μ_star[1]-1)
        end
    end
    n_star = hindex_n.(λ_max, μ_star)
    n_vals = union(_get_nvals_wavelet.(n_star)...)
    return n_vals
end

function _get_nvals(x, pf::ProjectedF{A, Tophat}) where A<:Union{Float64, Measurement}
    N_n = size(pf.fnlm)[1]
    n_star = [floor(Int, N_n*x)]

    if x ≈ 1.0
        n_star[1] -= 1
    elseif ((N_n*x)%1) == 0.0 && x ≉ 0.0
        push!(n_star, n_star[1]-1)
    end
    return n_star
end

"Evaluates the function f(uvec) by using fnlm coefficients and basis functions."
function (pf::ProjectedF{Float64,T})(uvec) where T<:Union{Wavelet, Tophat}
    u, θ, φ = uvec
    x = u/pf.radial_basis.umax
    n_vals = _get_nvals(x, pf)

    rad = VSDM.radRn.(n_vals, 0, u, (pf.radial_basis,))
    Y = VSDM.ylm_real.(pf.lm, θ, φ)
    basis_vals = Y' .* rad

    return dot(basis_vals, pf.fnlm[n_vals.+1, :])
end

function (pf::ProjectedF{A,B})(uvec) where {A<:Measurement, B<:Union{Wavelet, Tophat}}
    u, θ, φ = uvec
    x = u/pf.radial_basis.umax
    n_vals = _get_nvals(x, pf)

    rad = VSDM.radRn.(n_vals, 0, u, (pf.radial_basis,))
    Y = VSDM.ylm_real.(pf.lm, θ, φ)
    basis_vals = Y' .* rad

    fnlm = @view pf.fnlm[n_vals.+1, :]

    res = dot(basis_vals, Measurements.value.(fnlm))
    err_arr = basis_vals .* Measurements.uncertainty.(fnlm)
    err = sqrt(dot( err_arr, err_arr ))

    return (res ± err)
end

"Integral of (d^3 u) f^2(u) over the range a = [u_min, theta_min, phi_min]
to b = [u_max, theta_max, phi_max]."
function norm_energy(f::Function, a, b)
    return NIntegrate(x -> dV_sph(x)*f(x)^2, a, b, :cubature)
end

"Integral of (d^3 u) f^(u) for a ProjectedF is equal to 
sum_{nlm} f_{nlm}^2 * umax^3"
function norm_energy(pf::ProjectedF{Float64,T}) where T<:RadialBasis
    dot(pf.fnlm, pf.fnlm) * pf.radial_basis.umax^3
end

"If called with a ProjectedF{Measurement,B} will properly account for the
integration uncertainties."
function norm_energy(pf::ProjectedF{A,B}) where {A<:Measurement, B<:RadialBasis}
    vals = Measurements.value.(pf.fnlm)
    errs = Measurements.uncertainty.(pf.fnlm)

    res = dot(vals,vals)
    err_arr = @. abs(2*vals*errs)
    res_err = sqrt(dot(err_arr, err_arr))

    return (res ± res_err) * pf.radial_basis.umax^3
end

function norm_energy(fc::FCoeffs{Float64,T}) where T<:RadialBasis
    ff = collect(values(fc.fnlm))
    dot(ff,ff) * fc.radial_basis.umax^3
end

function norm_energy(fc::FCoeffs{A,B}) where {A<:Measurement, B<:RadialBasis}
    fnlm = collect(values(fc.fnlm))
    vals = Measurements.value.(fnlm)
    errs = Measurements.uncertainty.(fnlm)

    res = dot(vals, vals)
    err_arr = @. abs(2*vals*errs)
    res_err = sqrt(dot(err_arr, err_arr))

    return (res ± res_err) * fc.radial_basis.umax^3
end

function writeFnlm(outfile, fc::FCoeffs)
    vtype = valtype(fc.fnlm)

    rb_type = typeof(fc.radial_basis)
    rb_max = fc.radial_basis.umax

    nmax = maximum([k[1] for k in keys(fc.fnlm)])
    lmax = maximum([k[2] for k in keys(fc.fnlm)])

    q = collect(fc.fnlm)

    if vtype == Float64
        res = zeros(Float64, (length(q), 4))
        for i in eachindex(q)
            res[i,:] = [q[i][1]..., q[i][2]]
        end
    elseif vtype <: Measurement
        res = zeros(Float64, (length(q), 5))
        for i in eachindex(q)
            res[i,:] = [q[i][1]..., q[i][2].val, q[i][2].err]
        end
    end

    open(outfile, "w") do io
        write(io, "#, type: $rb_type, uMax: $rb_max, nMax: $nmax, ellMax: $lmax\n")
        if vtype == Float64
            write(io, "#, n, l, m, f\n")
        elseif vtype <: Measurement
            write(io, "#, n, l, m, f.val, f.err\n")
        end
        writedlm(io, res, ',')
    end

    return
end

function writeFnlm(outfile, pf::ProjectedF)
    n_vals = 0:(length(pf.fnlm[:,1])-1)
    nlm = [(n,lm...) for lm in pf.lm for n in n_vals]

    rb_type = typeof(pf.radial_basis)
    rb_max = pf.radial_basis.umax

    nmax = Int(maximum(n_vals))
    lmax = Int(maximum([lm[1] for lm in pf.lm]))

    open(outfile, "w") do io
        write(io, "#, type: $rb_type, uMax: $rb_max, nMax: $nmax, ellMax: $lmax\n")
        if eltype(pf.fnlm) == Float64
            res = hcat([[nlm[i]..., pf.fnlm[i]] for i in eachindex(nlm)]...)'
            write(io, "#, n, l, m, f\n")
            writedlm(io, res, ',')
        elseif eltype(pf.fnlm) <: Measurement
            res = hcat([[nlm[i]..., pf.fnlm[i].val, pf.fnlm[i].err] for i in eachindex(nlm)]...)'
            write(io, "#, n, l, m, f.val, f.err\n")
            writedlm(io, res, ',')
        end
    end

    return
end

function readFnlm(infile, radial_basis::RadialBasis; dict=true, use_err=true)
    input = readdlm(infile, ','; comments=true)
    nrow, ncol = size(input)
    if ncol == 4 || !use_err
        fnlm = Dict( (Int(input[i,1]), Int(input[i,2]), Int(input[i,3])) => 
                input[i,4] for i in 1:nrow )
    elseif ncol == 5
        fnlm = Dict( (Int(input[i,1]), Int(input[i,2]), Int(input[i,3])) => 
                (input[i,4] ± input[i,5]) for i in 1:nrow )
    end

    fc = FCoeffs(fnlm, radial_basis)
    if dict
        return fc
    else
        return ProjectedF(fc)
    end
end

function readFnlm(infile; dict=true, use_err=true)
    B = Dict{String,String}()
    open(infile) do io
        for l in eachline(io)
            if l[1] == '#'
                stuff = split.(split(filter(x -> !isspace(x), l[2:end]), ','), ':')
                filter!(x -> length(x) == 2, stuff)
                for s in stuff
                    B[s[1]] = s[2]
                end
                break
            end
        end
    end
    if B["type"] in ["Wavelet", "wavelet"]
        rb = Wavelet(parse(Float64, B["uMax"]))
    end

    return readFnlm(infile, rb; dict=dict, use_err=use_err)
end