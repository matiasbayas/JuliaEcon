# Neoclassical growth model solved with Chebyshev polynomial approximation

include("markov_approx.jl")

using Parameters, BasisMatrices

"""Parameters for the neoclassical growth model."""
@with_kw struct Params
    β::Float64 = 0.95
    α::Float64 = 0.4
    δ::Float64 = 0.05
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
end

"""Compute steady state capital stock."""
function SteadyState(z̄::Float64, p::Params)
    @unpack β, α, δ = p
    return ((1 / β - 1 + δ) / z̄ / α )^(1 / (α - 1))
end

function ProductivityProcess(p::Params)
    @unpack ρ, σ  = p
    z, p, Π = markov_tauchen(ρ, σ, N=3, m=2)
end

"""Get N Chebyshev nodes for grid [xlow, xhigh]."""
function cheb_nodes(xlow, xhigh, N)
    basis = Basis(ChebParams(N, xlow, xhigh))
    return nodes(basis)[1], basis
end

"""Chebyshev interpolation: given basis and data y, find polynomial coefficients."""
function cheb_interp(x, y, basis)
    Φ = BasisMatrix(basis, Direct(), x, 0)
    return Φ.vals[1] \ y
end

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
            return cheb_interp(k, c', basis), basis
        end
        c = cnew
    end
end

"""
    solveNeoclassical(p, N)

Solve the neoclassical growth model with N Chebyshev nodes.
Returns Chebyshev coefficients, steady state capital, and basis.
"""
function solveNeoclassical(p::Params, N)
    @unpack β, α, δ = p

    # get productivity process
    z, pr, Π = ProductivityProcess(p)
    z *= 0.3
    z̄ = pr ⋅ z

    # production and utility functions
    f(z, k) = z .* k .^ α .+ (1 - δ) .* k
    fk(z, k) =  α .* z .* k .^ (α - 1) .+ (1 - δ)
    up(c) = 1 / c
    up_inv(c) = 1 / c

    k_s = SteadyState(z̄, p)
    klow, khigh = 0.5*k_s, 2*k_s
    q, basis = ss_policy(klow, khigh, N, z, up, up_inv, f, fk, p::Params, Π)
    return q, k_s, basis
end
