using BasisMatrices

function cheb_nodes(xlow, xhigh, N)
    """ Returns basis object for Chebyshev interpolation """
    return Basis(ChebParams(N, xlow, xhigh))
end


function cheb_eval(x, q, basis)
    """ Evaluate chebyshev approximation in basis represented by q at points x """
    return BasisMatrix(basis, Direct(), [x]).vals[1]*q
end


function cheb_mat(basis; method = " Direct ")
    """ Get the matrix for chebyshev polynomial interpolation """
    if method == "Direct"
        return BasisMatrix(basis, Direct()).vals[1]
    elseif method == "Tensor"
        return BasisMatrix(basis, Tensor()).vals[1]
    end
end

function cheb_interp(y, Φ)
    """ Given n data points y and corresp. Vandermonde matrix of x's, finds interpolating Cheb. polynomial """
    Φ \ y
end

function cheb_der(q)
    """ Compute the derivative of the chebyshev polynomial q """

    if length(q) == 1
        return [0.]
    end

    n = length(q)
    der = zeros(n)
    for j in 1:n
        der[j-1] = 2*j*q[j]
        q[j-2] += (j*q[j])/(j-2)
    end
    if n > 1
        der[2] = 4*q[3]
    end
    der[1] = q[2]
    return der
end 


function cheb_newton(q, qder, x0, basis)
    for _ in 1:30
        y = cheb_eval(x0, q, basis)
        println(y)
        if abs(y) < 1E-14
            return x0
        end
        x0 -= y / cheb_eval(x0, qder, basis)
    end
end


