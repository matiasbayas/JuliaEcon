# General equilibrium incomplete markets model with transition dynamics

include("markov_approx.jl")
include("distribution.jl")
using Parameters, LinearAlgebra, Interpolations

"""Parameters for the incomplete markets model."""
@with_kw struct Params
    β::Float64 = 1. - 0.08/4
    ρ::Float64 = 0.975
    σ::Float64 = 0.7
end

"""Construct logspaced grid for assets, replicating np.logspace() behavior."""
function geomspace(amin::Float64, amax::Float64, N::Int64; pivot = 0.1)
    grid = 10 .^ (range(log10(amin+pivot), log10(amax+pivot),length=N)) .- pivot
    grid[1], grid[end] = amin, amax
    return grid
end

"""
    backward_iterate(c₊, a, y, r, r_post, Π, up, up_inv, p)

One step of backward iteration on the consumption policy function via EGM.
"""
function backward_iterate(c₊, a, y, r, r_post, Π, up, up_inv, p::Params)
    coh = (1 + r_post)*a .+ y'
    c_endog = up_inv( p.β * (1 + r) * up(c₊) * Π')
    c = similar(c_endog)
    for (s, yi) in enumerate(y)
        G = LinearInterpolation(c_endog[:, s] + a, c_endog[:, s],  extrapolation_bc = Line() )
        c[:, s] = G.(coh[:,s])
    end

    # deals with the constraint
    a₊ = coh - c
    a₊[a₊ .< a[1]] .= a[1]
    c = coh - a₊

    return c, a₊
end

"""Solve for steady state policy functions via backward iteration."""
function ss_policy(a, y, r, Π, up, up_inv, p::Params; maxit = 10000, tol = 1E-9, verbose = true)
    # initial guess for policy function
    c = 0.2 * ((1+r)*a .+ y')

    for it in 1:maxit
        c_new, a_new = backward_iterate(c, a, y, r, r, Π, up, up_inv, p)
        if mod(it, 10) == 0 && norm(c_new - c) < tol
            if verbose
                println("Convergence in $it iterations!")
            end
            return c_new, a_new
        end
        c = c_new
    end
end

"""
    ss(a, y, r, Π, up, up_inv, p)

Solve for the steady state: policy functions, ergodic distribution, and aggregates.
"""
function ss(a, y, r, Π, up, up_inv, p::Params; verbose = true)
    # solve for policy
    c, a₊ = ss_policy(a, y, r, Π, up, up_inv, p, verbose = verbose)

    # solve for distribution
    a₊i = Array{Int64}(undef, length(a), length(y))
    pi_a = Array{Float64}(undef, length(a), length(y))
    for i in 1:length(y)
        a₊i[:, i], pi_a[:, i] = interpolate_policy(a, a₊[:, i])
    end
    D = ergodic_dist(Π, a₊i, pi_a, verbose = verbose);

    # get aggregates
    C = c ⋅ D
    A = a ⋅ sum(D, dims = 2)

    return c, a₊, D, C, A
end

"""
    td_PE(a, y, r, Π, up, up_inv, p; rs, ys)

Compute partial equilibrium transition dynamics for the incomplete markets model.
Backward iterates on policy, then forward iterates on the distribution.
"""
function td_PE(a, y, r, Π, up, up_inv, p::Params; rs = nothing, ys = nothing)
    if ! isnothing(rs)
        T = length(rs)
    elseif ! isnothing(ys)
        T = size(ys)[1]
    end
    if isnothing(rs)
        rs = fill(r, T)
    end
    if isnothing(ys)
        ys = fill(y, T)
    end

    # computes initial and final steady state
    c, a₊, D, C, A = ss(a, y, r, Π, up, up_inv, p; verbose = false)
    # preallocate
    cs, a₊s, Ds = [Array{Float64}(undef, T, length(a), length(y)) for _ in 1:3]

    for t in reverse(1:T)
        if t == T
            c₊ = c
        else
            c₊ = cs[t+1,:,:]
        end
        if t == 1
            rlag = r
        else
            rlag = rs[t-1]
        end
        cs[t, :, :], a₊s[t, :, :] = backward_iterate(c₊, a, ys[t, :], rs[t], rlag, Π, up, up_inv, p)
    end

    Ds[1, :, :] = D
    for t in 1:T-1
        a₊i = Array{Int64}(undef, length(a), length(y))
        pi_a = Array{Float64}(undef, length(a), length(y))
        for i in 1:length(y)
            a₊i[:, i], pi_a[:, i] = interpolate_policy(a, a₊s[t, :, i])
        end
        Ds[t+1, :, :] = forward_iterate(Ds[t, :, :], Π, a₊i, pi_a)
    end

    # return aggregate paths
    return sum(Ds.*cs, dims = (2, 3)), sum(Ds.*a₊s, dims = (2, 3))
end
