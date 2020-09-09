using QuantEcon, Distributions, BasisMatrices, Parameters

function cheb_nodes(xlow, xhigh, N)
    basis = Basis(ChebParams(N, xlow, xhigh))
    return nodes(basis)[1], basis
end

function cheb_interp(x, y, basis)
    Φ = BasisMatrix(basis, Direct(), x, 0) # Direct() does better than Expanded()
    return Φ.vals[1] \ y
end


function cheb_interp2(y, basis)
    """ Efficient interpolation - input the basis from cheb_nodes and query points - y  """
    Φ = BasisMatrix(basis, Tensor()) # Direct() does better than Expanded() - try Tensor()
    return Φ.vals[1] \ y
end


function cheb_eval(x, q, basis)
    """ Evaluate chebyshev approximation represented by q at points x - not robust to case where x is scalar  """
    return BasisMatrix(basis, Expanded(), [x]).vals[1]*q
end

# Compute the $d^th$ order derivative of the Chebyshev representation:
function cheb_der(basis, x, d)
    Φ = BasisMatrix(basis, Direct(), x, d)
    return Φ.vals[1]
end

# Newton's method to find root, given initial guess x0 and basis we are working with
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

# brief check to see everythin is working
function newton(f, fder, x0)
    for _ in 1:30
        y = f(x0)
        if abs(y) < 1E-14
            return x0
        end
        x0 -= y / fder(x0)
    end
end

f(x) = exp(x) - 2
fder(x) = exp(x)

root = newton(f, fder, 0)
@assert root ≈ log(2)

# First step, find optimal adjsutment and switching thresholds:
function get_optimal_xs(qw, xi, c, basis, a, b)
    qw_der =  cheb_der(basis, xi, 1) 
    qw_2der = cheb_der(basis, xi, 2)
    
    xstar = cheb_newton(qw_der, qw_2der, 0., basis) # not sure if this is the root on [-1, 1]
    v_adj = cheb_eval(xstar, qw, basis) + c

    qw_root = qw 
    qw_root[1] -= v_adj

    # Now find lower and upper threshodls by starting at the opposite ends of state space
    # in theory, should check that this gives two roots that are ordered
    x_l = cheb_newton(qw_root, qw_der, a, basis)
    x_h = cheb_newton(qw_root, qw_der, b, basis)

    return xstar, x_l, x_h, v_adj
end

a = -2.
b = 2.
 
c = 1.

xi, basis = cheb_nodes(a, b, 15)
y = xi .^ 2
qw = cheb_interp(xi, y, basis)
qw_der =  cheb_der(basis, xi, 1) 
qw_2der = cheb_der(basis, xi, 2)

cheb_newton(qw_der, qw_2der, 0., basis)

# get_optimal_xs(qw, xi, c, basis, a , b)


# Second step, compute the expected values from adjustment and no adjustment










