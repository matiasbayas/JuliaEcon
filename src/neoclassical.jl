# Neoclassical growth model with discretized state space and backward iteration (EGM)

using LinearAlgebra, Distributions, Interpolations

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

"""Returns variance of discretized random variable with support x and pmf p."""
function variance(x, p)
    return p ⋅ (x .- p ⋅ x) .^ 2
end

"""
    markov_tauchen(ρ, σ; N=7, m=3)

Tauchen method discretizing AR(1) s_t = ρ * s_{t-1} + ϵ_t.
Returns `(y, p, Π)`: states, stationary distribution, and transition matrix.
"""
function markov_tauchen(ρ, σ; N=7, m = 3)
    s = range(-m, m, length = N)
    ds = s[2] - s[1]
    sd_innov = sqrt(1 - ρ ^ 2)

    Π = Array{Float64}(undef, N, N)
    Π[:, 1] = cdf.(Normal(0.0, sd_innov), s[1] .- ρ * s .+ ds / 2)
    Π[:, end] = 1 .- cdf.(Normal(0.0, sd_innov), s[end] .- ρ * s .- ds / 2)
    for j in 2:N-1
        Π[:, j] = cdf.(Normal(0.0, sd_innov), s[j] .- ρ * s .+ ds / 2) - cdf.(Normal(0.0, sd_innov), s[j] .- ρ * s .- ds / 2)
    end

    p = stationary(Π)
    s *= ( σ / sqrt(variance(s, p)))
    y = exp.(s) ./ ( p ⋅ exp.(s))

    return y, p, Π
end

f(z, k) = z .* k .^ α .+ (1 - δ) .* k
fk(z, k) =  α .* z .* k .^ (α - 1) .+ (1 - δ)
up(c) = 1 / c
up_inv(c) = 1 / c

function backward_iterate(cplus, k)
    c_endog = up_inv.( β * Π * ( fk(z, k') .* up.(cplus)))
    c = similar(c_endog)
    for (i, zi) in enumerate(z)
        G = LinearInterpolation(c_endog[i, :] + k, c_endog[i, :], extrapolation_bc = Line())
        c[i, :] = G.(f(zi, k))
    end
    return c
end

z, p, Π = markov_tauchen(0.9, 0.2, N = 3, m = 2);
z *= 0.3;
α = 0.4;
δ = 0.05;
β = 0.95;

z̄ = p ⋅ z;
kstar = ((1 / β - 1 + δ ) / z̄ / α ) ^ (1 / (α - 1));

k = range(0.5*kstar, 2*kstar, length = 300);

function ss_policy(k; maxit = 10_000, tol = 1E-10)
    c = 0.3 * repeat(k', 3, 1)
    for it in 1:maxit
        cnew = backward_iterate(c, k)
        if mod(it, 10) ≈ 0 && norm(cnew - c) < tol
            return cnew
        end
        c = cnew
    end
end
