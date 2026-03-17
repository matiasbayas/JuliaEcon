# Chebyshev polynomial interpolation and root-finding utilities using BasisMatrices.jl

using BasisMatrices

"""Returns basis object for Chebyshev interpolation."""
function cheb_nodes(xlow, xhigh, N)
    return Basis(ChebParams(N, xlow, xhigh))
end

"""Evaluate Chebyshev approximation in basis represented by q at points x."""
function cheb_eval(x, q, basis)
    return BasisMatrix(basis, Direct(), [x]).vals[1]*q
end

"""Get the matrix for Chebyshev polynomial interpolation."""
function cheb_mat(basis; method = "Direct")
    if method == "Direct"
        return BasisMatrix(basis, Direct()).vals[1]
    elseif method == "Tensor"
        return BasisMatrix(basis, Tensor()).vals[1]
    end
end

"""Given n data points y and corresponding Vandermonde matrix of x's, finds interpolating Chebyshev polynomial."""
function cheb_interp(y, Φ)
    return Φ \ y
end

"""Newton's method root-finding using Chebyshev polynomial representations."""
function cheb_newton(q, qder, x0, basis)
    for _ in 1:30
        y = cheb_eval(x0, q, basis)
        if abs(y) < 1E-14
            return x0
        end
        x0 -= y / cheb_eval(x0, qder, basis)
    end
end
