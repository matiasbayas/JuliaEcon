using QuantEcon, Distributions, BasisMatrices, Parameters, Expectations

# chebyshev block
function cheb_nodes(xlow, xhigh, N)
    """ Gets Chebyshev nodes  """
    return Basis(ChebParams(N, xlow, xhigh))
end

function cheb_interp(y, basis)
    """ Efficient interpolation - input the basis from cheb_nodes and query points - y  """
    Φ = BasisMatrix(basis, Tensor()) # Direct() does better than Expanded() - try Tensor()
    return Φ.vals[1] \ y
end

function cheb_eval(x, q, basis)
    """ Evaluate chebyshev approximation represented by q at points x - not robust to case where x is scalar  """
    return BasisMatrix(basis, Expanded(), [x]).vals[1]*q
end

function cheb_mat(basis; method = " Direct ")
    """ Get the matrix for chebyshev polynomial interpolation on the grid """
    if method == "Direct"
        return BasisMatrix(basis, Direct()).vals[1]
    elseif method == "Tensor"
        return BasisMatrix(basis, Tensor()).vals[1]
    end
end

function cheb_interp2(y, Φ)
    Φ \ y
end

# Compute the $d^th$ order derivative of the Chebyshev representation:
function cheb_der(basis, x, d)
    Φ = BasisMatrix(basis, Direct(), x, d)
    return Φ.vals[1]
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

a = -2.
b = 2.
basis = cheb_nodes(a, b, 15)
xi = nodes(basis)[1]
y = xi .^ 2
qw = cheb_interp(y, basis)
qw_der =  cheb_interp2(cheb_der(basis, xi, 1) 
qw_2der = cheb_der(basis, xi, 2)



