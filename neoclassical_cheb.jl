include("markov_approx.jl")

using Parameters, BasisMatrices, BenchmarkTools

# Define class with std parameters for the model
@with_kw struct Params
    β::Float64 = 0.95
    α::Float64 = 0.4
    δ::Float64 = 0.05
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
end

# Function to compute steady state of capital
function SteadyState(z̄::Float64, p::Params)
    @unpack β, α, δ = p
    k_s = ((1 / β - 1 + δ) / z̄ / α )^(1 / (α - 1))
end


function ProductivityProcess(p::Params)
    @unpack ρ, σ  = p
    z, p, Π = markov_tauchen(ρ, σ, N=3, m=2)
end

# automate chebyshev stuff - can still be improved
function cheb_nodes(xlow, xhigh, N)
    basis = Basis(ChebParams(N, xlow, xhigh))
    return nodes(basis)[1], basis
end

function cheb_interp(x, y, basis)
    Φ = BasisMatrix(basis, Direct(), x, 0)
    return Φ.vals[1] \ y
end

# main iteration functions

function backward_iterate(cplus, k, z, up, up_inv, f, fk, basis, p::Params, Π)
    c_endog = up_inv.( p.β * Π * ( fk(z, k') .* up.(cplus)))
    c = similar(c_endog)
    for (i, zi) in enumerate(z)
        q = cheb_interp( c_endog[i, :] + k, c_endog[i, :], basis)
        c[i, :] = BasisMatrix(basis, Direct(), f(zi, k)).vals[1]*q
    end
    return c
end

function ss_policy(klow, khigh, N, z, up, up_inv, f, fk, p::Params, Π; maxit = 10_000, tol = 1E-10)
    k, basis = cheb_nodes(klow, khigh, N)
    c = 0.3 * repeat(k', 3, 1)
    for it in 1:maxit
        cnew = backward_iterate(c, k, z, up, up_inv, f, fk, basis, p::Params, Π)
        if mod(it, 10) ≈ 0 && norm(cnew - c) < tol
            #println("Convergence in $it iterations!")
            return cheb_interp(k, c', basis), basis
        end
        c = cnew
    end
end

function solveNeoclassical(p::Params, N)
    @unpack β, α, δ = p

    # get productivity process
    z, pr, Π = ProductivityProcess(p)
    z *= 0.3
    z̄ = pr ⋅ z

    # production and utility functions
    f(z, k) = z .* k .^ α .+ (1 - δ) .* k
    fk(z, k) =  α .* z .* k .^ (α - 1) .+ (1 - δ) ;
    up(c) = 1 / c
    up_inv(c) = 1 / c

    k_s = SteadyState(z̄, p)
    klow, khigh = 0.5*k_s, 2*k_s
    q, basis = ss_policy(klow, khigh, N, z, up, up_inv, f, fk, p::Params, Π)
    return q, k_s, basis
end

@btime solveNeoclassical(Params(), 300, 10);
q, k_s, basis = solveNeoclassical(Params(), 300, 10);

using Plots
k = kGrid(k_s, 300);
c = funeval(q, basis, k);
plot(k, c, label = ["z_low" "z_mid" "z_high"], lw = 2, legend=:topleft);
