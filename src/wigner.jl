
function _Gl_from_Dl!(Gl,Dl,ell)
    for mp in 1:ell
        for m in 1:ell
            d_mp = (-1)^m * Dl[ell-mp+1,ell+m+1]
            d_pp = (-1)^(mp-m) * Dl[ell+mp+1,ell+m+1]
            Gl[ell-mp+1, ell-m+1] = real(d_pp) - real(d_mp)
            Gl[ell-mp+1, ell+m+1] = -imag(d_pp) + imag(d_mp)
            Gl[ell+mp+1, ell-m+1] = imag(d_pp) + imag(d_mp)
            Gl[ell+mp+1, ell+m+1] = real(d_pp) + real(d_mp)
        end
        d_p0 = sqrt(2) * (-1)^mp * Dl[ell+mp+1, ell+1]
        Gl[ell-mp+1, ell+1] = -imag(d_p0)
        Gl[ell+mp+1, ell+1] = real(d_p0)
    end

    for m in 1:ell
        d_0p = sqrt(2) * (-1)^m * Dl[ell+1, ell+m+1]
        Gl[ell+1, ell-m+1] = imag(d_0p)
        Gl[ell+1, ell+m+1] = real(d_0p)
    end

    Gl[ell+1, ell+1] = real(Dl[ell+1, ell+1])
end

"Can be called with a pre-computed D matrix instead (see SphericalFunctions.jl)"
function G_matrices(D_matrices, ell_max)
    G = zeros(Float64, size(D_matrices))
    for (ell, Dl, Gl) in zip(0:ell_max, D_iterator(D_matrices, ell_max),
                             D_iterator(G, ell_max))
        g = transpose(Gl)
        d = conj(transpose(Dl))
        _Gl_from_Dl!(g, d, ell)
    end
    return G
end

"""
    G_matrices(R::T, ell_max) where T<:Union{Quaternion,Rotor}

Calculates the analog of the Wigner D matrices for real spherical harmonics,
which we refer to as the "G matrices" for a given rotation represented as a 
quaternion (or rotor). If you'd like to use other rotations, see the relevant
conversion functions in Quaternionic.jl
"""
function G_matrices(R::T, ell_max) where T<:Union{Quaternion,Rotor}
    D = D_matrices(R, ell_max)
    return G_matrices(D, ell_max)
end

function G_matrices!(G, D::Vector, ell_max)
    for (ell, Dl, Gl) in zip(0:ell_max, D_iterator(D, ell_max),
                             D_iterator(G, ell_max))
        g = transpose(Gl)
        d = conj(transpose(Dl))
        _Gl_from_Dl!(g,d,ell)
    end
end

function G_matrices!(G, D::Tuple)
    ell_max = D[2]
    for (ell, Dl, Gl) in zip(0:ell_max, D_iterator(D[1], ell_max),
                             D_iterator(G, ell_max))
        g = transpose(Gl)
        d = conj(transpose(Dl))
        _Gl_from_Dl!(g,d,ell)
    end
end