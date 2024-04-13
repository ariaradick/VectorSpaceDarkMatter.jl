function hindex_n(L::Int, M::Int)
    if L==-1
        return zero(L)
    end
    return 2^L + M
end

function hindex_LM(n::Int)
    if n==0
        return [-one(n), zero(n)]
    end
    L = convert(typeof(n), floor(log2(n)))
    M = n - 2^L
    return [L,M]
end

function haar_x123(n::Int)
    if n==0
        return [0.0, 0.5, 1.0]
    end
    L,M = hindex_LM(n)
    x1 = 2.0^(-L) * M
    x2 = 2.0^(-L) * (M+0.5)
    x3 = 2.0^(-L) * (M+1.0)
    return [x1, x2, x3]
end

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

function _bin_integral(x1, x2, dim=3)
    (x2^dim - x1^dim)/dim
end

function _haar_sph_integral(n::Int, dim=3)
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