"""Gives the n that corresponds to a given L and M value, from n = 2^L + M (L=-1 -> n=0)"""
function hindex_n(L::Int, M::Int)
    if L==-1
        return zero(L)
    end
    return 2^L + M
end

"""Gives the L and M that corresponds to a given n value, from n = 2^L + M (n=0 -> L=-1)"""
function hindex_LM(n::Int)
    if n==0
        return [-one(n), zero(n)]
    end
    L = convert(typeof(n), floor(log2(n)))
    M = n - 2^L
    return [L,M]
end

"""Base of support of nth wavelet, including midpoint."""
function haar_x123(n::Int)
    if n==0
        return [0.0, 1.0]
    end
    L,M = hindex_LM(n)
    x1 = 2.0^(-L) * M
    x2 = 2.0^(-L) * (M+0.5)
    x3 = 2.0^(-L) * (M+1.0)
    return [x1, x2, x3]
end

"""Base of support of nth wavelet, not including midpoint."""
function _haar_x13(n::Int)
    if n==0
        return [0.0, 1.0]
    end
    L,M = hindex_LM(n)
    x1 = 2.0^(-L) * M
    x3 = 2.0^(-L) * (M+1.0)
    return [x1, x3]
end

""" Returns the value of h_n(x) where it is non-zero. """
function haar_sph_value(n::Int; dim=3)
    if n == 0
        return sqrt(dim)
    end
    x1, x2, x3 = haar_x123(n)
    y1 = x1^dim
    y2 = x2^dim
    y3 = x3^dim
    A = sqrt(dim/(y3 - y1) * (y3-y2)/(y2-y1))
    B = sqrt(dim/(y3 - y1) * (y2-y1)/(y3-y2))
    return [A,-B]
end

"""
    _bin_integral(x1, x2; dim=3)

Integral of int(x**(dim-1) dx) on the interval [x1, x2], where dim is the
number of dimensions. Needed for haar_transform.
    This is <1|bin_n>, for unnormalized 'bin_n(x) = 1 iff x in [x1,x2]'.
"""
function _bin_integral(x1, x2; dim=3)
    (x2^dim - x1^dim)/dim
end

"""
    _haar_sph_integral(n::Int; dim=3)

Returns the integrals int(x**(dim-1)dx h_n(x)) on the regions A and B.

Volume integrals A and B are equal in magnitude.
"""
function _haar_sph_integral(n::Int; dim=3)
    if n == 0
        return 1 / sqrt(n)
    end
    x1, x2, x3 = haar_x123(n)
    y1 = x1^dim
    y2 = x2^dim
    y3 = x3^dim
    integralAB = sqrt((y2-y1)*(y3-y2)/(dim*(y3-y1)))
    return [integralAB, -integralAB]
end

"""Returns the value of h_{L,M}(x) where it is nonzero, for n>0."""
function _haar_sph_value_LM(L, M; dim=3)
    x1 = 2.0^(-L) * M
    x2 = 2.0^(-L) * (M+0.5)
    x3 = 2.0^(-L) * (M+1)
    y1 = x1^dim
    y2 = x2^dim
    y3 = x3^dim
    A = sqrt(dim/(y3 - y1) * (y3-y2)/(y2-y1))
    B = sqrt(dim/(y3 - y1) * (y2-y1)/(y3-y2))
    return [A,-B]
end

"""Normalized spherical Haar wavelet, n=0,1,2,..."""
function haar_fn_x(n, x; dim=3)
    hval = haar_sph_value(n, dim=dim)

    if n == 0
        if 0 <= x <= 1
            return hval
        # elseif x==0 || x==1
        #     return 0.5*hval
        else
            return 0.0
        end
    else
        x1,x2,x3 = haar_x123(n)
        if x1 < x < x2
            return hval[1]

        elseif x2 < x < x3
            return hval[2]

        elseif x==x1
            if x==0.0
                return hval[1]
            else
                return 0.5*(hval[1])
            end

        elseif x==x2
            return 0.5*sum(hval)

        elseif x==x3
            if x == 1.0
                return hval[2]
            else
                return 0.5*(hval[2])
            end
            
        else
            return 0.0
        end
    end
end

"""'True' if nd is a descendant of 'n', 'False' otherwise."""
function _h_n_covers_nd(n, nd)
    if n >= nd
        return False
    end
    # else:
    while nd > 1
        nd = floor(Int, nd/2)
        if nd==n
            return True
        end
    end
    return False
end

function haar_inverse(f_n_list; dim=3)
    nGens = ceil(Int, log2(length(f_n_list)))
    nCoeffs = 2^nGens
    Lmax = nGens - 1
    f_x = ones(nCoeffs)*f_n_list[1]*haar_sph_value(0, dim=dim)

    for L in 0:Lmax
        w_L = Vector{eltype(f_n_list)}()
        for M in 0:(2^L-1)
            A_n, mB_n = _haar_sph_value_LM(L, M, dim=dim)
            n = hindex_n(L,M)
            f_n = f_n_list[n+1]
            bin_size = 2^(Lmax-L)
            append!(w_L, ones(typeof(f_n), bin_size) * A_n * f_n)
            append!(w_L, ones(typeof(f_n), bin_size) * mB_n * f_n)
        end
        f_x[:] += w_L[:]
    end
    return f_x
end