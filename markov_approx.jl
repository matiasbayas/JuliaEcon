""" Methods to discretize AR(1) processes - based on Rognlie's material """


using LinearAlgebra, Distributions

function stationary(Π; p_seed = nothing, tol = 1E-11, maxit = 10_000)

    """ Find invariant distribution of Markov chain by iteration """

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
end;

function variance(x, p)
    """ Returns variance of discretized rv with support x and probability mass function p"""
    return p ⋅ (x .- p ⋅ x) .^ 2
end;

function markov_tauchen(ρ, σ; N=7, m = 3)
    """Tauchen method discretizing AR(1) s_t = ρ * s_{t-1} + ϵ_t.
    Parameters
    ----------
    ρ   : scalar, persistence
    σ   : scalar, unconditional sd of s_t
    N   : int, number of states in discretized Markov process
    m   : scalar, discretized s goes from approx -m*sigma to m*sigma

    Returns
    ----------
    y  : array (N), states proportional to exp(s) s.t. E[y] = 1
    p : array (N), stationary distribution of discretized process
    Π : array (N*N), Markov matrix for discretized process
    """
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
end;

function rouwenhorst(ρ, σ; N=7)
    """ Rouwenhorst method to discretize AR(1) process

    Parameters
    -----------
    ρ  :  float, persistence
    σ  :  float, standard deviation of innovations
    N  :  int, numer of states in discretized process

    Returns
    ---------
    s  : array(n), values at N discretized states
    pr : array(n), stationary distribution across N states
    Π  : array(n*n), transition matrix
    """

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
        Π[2:end-1, :] /= 2   #why do you do this??
    end

    pr = stationary(Π)
    s = collect(range(-1., 1., length = N))
    s *= σ / sqrt(variance(s, pr))
    y = exp.(s) ./ ( pr ⋅ exp.(s))

    return y, pr, Π
end
