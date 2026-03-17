# Neoclassical growth model with endogenous labor via Chebyshev polynomials

include("markov_approx.jl")

using Parameters, BasisMatrices, QuantEcon

"""Parameters for the neoclassical model with endogenous labor."""
@with_kw struct Params
    β::Float64 = 0.95
    α::Float64 = 0.4
    δ::Float64 = 0.05
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
end

"""Compute steady state capital, labor, and consumption given parameters and production function F."""
function SteadyState(p::Params, F)
    @unpack β, α, δ = p
    k_n = ((1.0 / β - 1.0 + δ) / α )^(1.0 / (α - 1.0))
    D1 = (1.0-α) * k_n ^ α
    D2 = F(1.0, k_n, 1.0) - k_n
    cstar = sqrt(D1 * D2)
    nstar = cstar / D2
    kstar = k_n * nstar
    return cstar, nstar, kstar
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
function cheb_interp(y, basis)
    Φ = BasisMatrix(basis, Tensor())
    return Φ.vals[1] \ y
end

"""
    FOC(n, z_cur, k_cur, z, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p, pr, basis)

Euler equation residual for the labor choice. Used as the objective for
root-finding (Brent's method) at each grid point.
"""
function FOC(n::Float64, z_cur::Float64, k_cur::Float64, z, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p::Params, pr, basis)
    up_c = vp(n) / Fn(z_cur, k_cur, n)
    c = up_inv(up_c)
    k₊ = F(z_cur, k_cur, n) - c
    if k₊ < 0.
        return -1E-6
    end
    c₊ = BasisMatrix(basis, Direct(), [k₊] ).vals[1]*cq₊
    n₊ = BasisMatrix(basis, Direct(), [k₊] ).vals[1]*nq₊
    inside_E = Fk(z', k₊, n₊) .* up(c₊)
    return up_c .- p.β * (inside_E ⋅ pr)
end

function backward_iterate(k, z, nmin, nmax, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis)
    nq = similar(nq₊)
    cq = similar(cq₊)
    for (i, z_cur) in enumerate(z)
        pr_z = Π[i, :]
        n = similar(k)
        c = similar(k)
        for (j, k_cur) in enumerate(k)
            h(ni) = FOC(ni, z_cur, k_cur, z, nq₊, cq₊, up, up_inv, vp, F, Fn, Fk, p::Params, pr_z, basis)
            n[j] = brent(h, nmin, nmax)
            c[j] = up_inv(vp(n[j]) / Fn(z_cur, k_cur, n[j]))
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
            return nq, cq
        end
        nq = nq_new
        cq = cq_new
    end
end

"""
    solveNeoclassical(p, N)

Solve the neoclassical model with endogenous labor supply using N Chebyshev nodes.
Returns Chebyshev coefficients for labor and consumption policies.
"""
function solveNeoclassical(p::Params, N)
    @unpack β, α, δ = p

    # get productivity process
    z, pr, Π = ProductivityProcess(p)

    # production and utility functions
    F(z, k, n) = z .* k.^α .* n.^(1.0 - α) .+ (1.0 - δ) .* k
    Fk(z, k, n) =  α .* z .* (n ./ k).^(1.0 - α) .+ (1.0 - δ)
    Fn(z, k, n) = (1.0 - α) .* z .* (k ./ n).^α
    up(c) = 1.0 ./ c
    up_inv(c) = 1.0 ./ c
    vp(n) = n # Frisch elasticity of one

    c_s, n_s, k_s = SteadyState(p, F)
    klow, khigh = 0.4 * k_s, 2.5 * k_s
    k, basis = cheb_nodes(klow, khigh, N)
    nmin, nmax = 0.2 * n_s, 6.0 * n_s
    nq, cq = ss_policy(k, z, nmin, nmax, c_s, k_s, n_s, up, up_inv, vp, F, Fn, Fk, p::Params, Π, basis)
    return nq, cq, k, basis
end
