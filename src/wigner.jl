
function _Glmpm_from_Dl(Dl, ell, mp, m)
    if mp < 0 && m < 0
        return real(Dl[ell-mp+1, ell-m+1] - (-1)^mp * Dl[ell+mp+1, ell-m+1])
    elseif mp < 0 && m == 0
        return -sqrt(2)*imag(Dl[ell-mp+1, ell+1])
    elseif mp < 0 && m > 0
        return -imag(Dl[ell-mp+1, ell+m+1] - (-1)^mp * Dl[ell+mp+1, ell+m+1])
    elseif mp == 0 && m < 0
        return sqrt(2)*imag(Dl[ell+1, ell-m+1])
    elseif mp==0 && m==0
        return real(Dl[ell+1, ell+1])
    elseif mp==0 && m>0
        return sqrt(2)*real(Dl[ell+1, ell+m+1])
    elseif mp>0 && m<0
        return imag(Dl[ell+mp+1, ell-m+1] + (-1)^mp * Dl[ell-mp+1, ell-m+1])
    elseif mp>0 && m==0
        return sqrt(2)*real(Dl[ell+mp+1, ell+1])
    elseif mp>0 && m>0
        return real(Dl[ell+mp+1, ell+m+1] + (-1)^mp * Dl[ell-mp+1, ell+m+1])
    end
end

function G_matrices(D_matrices, ell_max)
    G = zeros(Float64, size(D_matrices))
    i = 1
    for (ell, Dl) in zip(0:ell_max, D_iterator(D_matrices, ell_max))
        for mp in -ell:ell
            for m in -ell:ell
                G[i] = _Glmpm_from_Dl(Dl, ell, mp, m)
                i += 1
            end
        end
    end
    return G
end

function G_matrices(R::T, ell_max) where T<:Union{Quaternion,Rotor}
    D = D_matrices(R, ell_max)
    return G_matrices(D, ell_max)
end

function G_matrices(α::Real, β::Real, γ::Real, ell_max)
    Rq = from_euler_angles(α,β,γ)
    G_matrices(Rq, ell_max)
end

function G_matrices!(G, D::Vector, ell_max)
    i = 1
    for (ell, Dl) in zip(0:ell_max, D_iterator(D, ell_max))
        for mp in -ell:ell
            for m in -ell:ell
                G[i] = _Glmpm_from_Dl(Dl, ell, mp, m)
                i += 1
            end
        end
    end
end

function G_matrices!(G, D::Tuple)
    ell_max = D[2]
    i = 1
    for (ell, Dl) in zip(0:ell_max, D_iterator(D[1], ell_max))
        for mp in -ell:ell
            for m in -ell:ell
                G[i] = _Glmpm_from_Dl(Dl, ell, mp, m)
                i += 1
            end
        end
    end
end