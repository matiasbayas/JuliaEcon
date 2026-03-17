# Tauchen and Rouwenhorst methods to discretize AR(1) processes

using LinearAlgebra, Distributions

"""Find invariant distribution of Markov chain by iteration."""
function stationary(Π; p_seed = nothing, tol = 1E-11, maxit = 10_000)
    if isnothing(p_seed)
        p = ones(1, size(Π)[1]) / size(Π)[1]
    else
        p = p_seed
    end

    for it in 1:maxit
        p_new = p * Π
        if norm(p_new - p) < tol
            break
        end
        p = p_new

        if it == maxit
            println("No convergence after $maxit iterations!")
        end
    end
    return p
end

"""Returns variance of discretized random variable with support x and probability mass function p."""
function variance(x, p)
    return p ⋅ (x .- p ⋅ x) .^ 2
end

"""
    markov_tauchen(ρ, σ; N=7, m=3)

Tauchen method discretizing AR(1) s_t = ρ * s_{t-1} + ϵ_t.

Returns `(y, p, Π)`: states proportional to exp(s) with E[y]=1,
stationary distribution, and transition matrix.
"""
function markov_tauchen(ρ, σ; N=7, m = 3)
    # make normalized grid, start with cross-sectional sd of 1
    s = range(-m, m, length = N)
    ds = s[2] - s[1]
    sd_innov = sqrt(1 - ρ ^ 2)

    # standard Tauchen method to generate Π given N and m
    Π = Array{Float64}(undef, N, N)
    Π[:, 1] = cdf.(Normal(0.0, sd_innov), s[1] .- ρ * s .+ ds / 2)
    Π[:, end] = 1 .- cdf.(Normal(0.0, sd_innov), s[end] .- ρ * s .- ds / 2)
    for j in 2:N-1
        Π[:, j] = cdf.(Normal(0.0, sd_innov), s[j] .- ρ * s .+ ds / 2) - cdf.(Normal(0.0, sd_innov), s[j] .- ρ * s .- ds / 2)
    end

    # invariant distribution and scaling
    p = stationary(Π)
    s *= ( σ / sqrt(variance(s, p)))
    y = exp.(s) ./ ( p ⋅ exp.(s))

    return y, p, Π
end

"""
    markov_rouwenhorst(ρ, σ; N=7)

Rouwenhorst method to discretize AR(1) process.

Returns `(y, pr, Π)`: states, stationary distribution, and transition matrix.
"""
function markov_rouwenhorst(ρ, σ; N=7)
    # parametrize Rouwenhorst markov matrix for n=2
    p = (1. + ρ) / 2
    Π = [p 1-p; 1-p p]

    # implement recursion to build from n=3 to n=N
    for n in 3:N
        Π_old = Π
        Π = zeros(n, n)
        Π[1:end-1, 1:end-1] += p * Π_old
        Π[1:end-1, 2:end] += (1-p) * Π_old
        Π[2:end, 1:end-1] += (1-p) * Π_old
        Π[2:end, 2:end] += p * Π_old
        Π[2:end-1, :] /= 2
    end

    pr = stationary(Π)
    s = collect(range(-1., 1., length = N))
    s *= σ / sqrt(variance(s, pr))
    y = exp.(s) ./ ( pr ⋅ exp.(s))

    return y, pr, Π
end
