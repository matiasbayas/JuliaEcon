""" Solve neoclassical model with endogenous labor via Chebyshev polynomials """

include("markov_approx.jl")

using Parameters, BasisMatrices, BenchmarkTools, QuantEcon

# Define class with std parameters for the model
@with_kw struct Params
    β::Float64 = 0.95
    α::Float64 = 0.4
    δ::Float64 = 0.05
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
end

# Function to compute steady state of capital, labor, and consumption
function SteadyState(p::Params, F)
    """ Inputs - parameters and production function F """
    @unpack β, α, δ = p
    k_n = ((1.0 / β - 1.0 + δ) / α )^(1.0 / (α - 1.0))
    D1 = (1.0-α) * k_n ^ α
    D2 = F(1.0, k_n, 1.0) - k_n
    cstar = sqrt(D1 * D2)
    nstar = cstar / D2
    kstar = k_n * nstar
    return cstar, nstar, kstar
end

# Function to discretize AR(1) process using Rognlie's routines
function ProductivityProcess(p::Params)
    @unpack ρ, σ  = p
    z, p, Π = markov_tauchen(ρ, σ, N=3, m=2)
end

# automate chebyshev stuff - can still be improved
function cheb_nodes(xlow, xhigh, N)
    """ Get N chebyshev nodes for grid [xlow, xhigh] """
    basis = Basis(ChebParams(N, xlow, xhigh))
    return nodes(basis)[1], basis
end

function cheb_interp(y, basis)
    """ Efficient interpolation - input the basis from cheb_nodes and query points - y  """
    Φ = BasisMatrix(basis, Tensor()) # Direct() does better than Expanded() - try Tensor()
    return Φ.vals[1] \ y
end

# Assume nq+ and cq+ are chebyshev representations of n+ and c+
# pr is the vector giving probability of transitioning to each future z, given current z
function FOC(n::Float64, z::Float64, k::Float64, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p::Params, pr, basis)
    """ EE to find root with respect to labor choice - n
        Takes as inputs:
        1. guess for today's optimal labor supply - n (scalar - this is what we will find zero over)
        2. capital stock today - k (scalar)
        3. level of productivity - z (scalar)
        4. tomorrow's policy functions in chebyshev form - nq+, cq+
        5. prod fun utility fun and their derivatives/inverses - up, up_inv, F,  F_n, F_k
        6. parameters for the model - p
        7. transition probabilities - pr
        8. basis in which we are interpolating - basis """
    up_c = vp(n) / Fn(z, k, n)
    c = up_inv(up_c)
    k₊ = F(z, k, n) - c
    if k₊ < 0.
        return -1E-6
    end
    #c₊ = funeval(cq₊, basis, k₊)
    #n₊ = funeval(nq₊, basis, k₊)
    c₊ = BasisMatrix(basis, Direct(), [k₊] ).vals[1]*cq₊  # make sure to pass [k] not just k - does not work with float
    n₊ = BasisMatrix(basis, Direct(), [k₊] ).vals[1]*nq₊
    inside_E = Fk(z, k₊, n₊) .* up(c₊)
    return up_c .- p.β * (pr ⋅ inside_E)
end

function backward_iterate(k, z, nmin, nmax, nq₊, cq₊,up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis)
    nq = similar(nq₊)
    cq = similar(cq₊)
    for (i, z) in enumerate(z)
        #print(i)
        pr = Π[i, :]
        n = similar(k)
        c = similar(k)
        for (j, k) in enumerate(k)
            h(ni) = FOC(ni, z, k, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p::Params, pr, basis)
            n[j] = brent(h, nmin, nmax) #this does a lot better than the f_zero from roots
            c[j] = up_inv(vp(n[j]) / Fn(z, k, n[j]))
        end
        nq[:, i] = cheb_interp(n, basis)
        cq[:, i] = cheb_interp(c, basis)
    end
    return nq, cq
end

function ss_policy(k, z, nmin, nmax, c_s, k_s, n_s, up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis; maxit = 200, tol = 5E-9)
    c = repeat(c_s ./ k_s .* k', length(z), 1)
    n = repeat(n_s .* z, 1, length(k))
    cq = cheb_interp(c', basis)
    nq = cheb_interp(n', basis)
    for it in 1:maxit
        nq_new, cq_new = backward_iterate(k, z, nmin, nmax, nq, cq, up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis)
        if mod(it, 10) ≈ 0 && norm(nq_new - nq) < tol
            #println("Convergence in $it iterations!")
            return nq, cq
        end
        nq = nq_new
        cq = cq_new
    end
end

function solveNeoclassical(p::Params, N)
    @unpack β, α, δ = p

    # get productivity process
    z, pr, Π = ProductivityProcess(p)

    # production and utility functions
    F(z, k, n) = z .* k.^α .* n.^(1.0 - α) .+ (1.0 - δ) .* k
    Fk(z, k, n) =  α .* z .* (n ./ k).^(1.0 - α) .+ (1.0 - δ)
    Fn(z, k, n) = (1.0 - α) .* z .* (k ./ n).^α
    up(c) = 1.0 ./ c
    up_inv(c) = 1.0 / c #this doesn't really have to be broadcasted bc it only takes scalars as inputs
    vp(n) = n # frisch elasticity of one

    c_s, n_s, k_s = SteadyState(p, F)
    klow, khigh = 0.4 * k_s, 2.5 * k_s
    k, basis = cheb_nodes(klow, khigh, N)
    nmin, nmax = 0.2 * n_s, 6.0 * n_s
    nq, cq = ss_policy(k, z, nmin, nmax, c_s, k_s, n_s, up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis)
    return nq, cq, k, basis
end

#@btime ProductivityProcess(Params())
#@btime solveNeoclassical(Params(), 15);

# Profile the code
#using Profile
#Profile.clear();
#Profile.init(delay = 0.05)
#@profile solveNeoclassical(Params(), 15)
#Profile.print(format=:flat)

#nq, cq, k, basis= solveNeoclassical(Params(), 15);





#using Plots
#c = funeval(cq, basis, k);
#n = funeval(nq, basis, k);
#plot(k, c, label = ["z_low" "z_mid" "z_high"], lw = 2, legend=:topleft)
#plot(k, n, label = ["z_low" "z_mid" "z_high"], lw = 2, legend=:topleft)
