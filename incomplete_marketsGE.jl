include("markov_approx.jl")
include("distribution.jl")
using Parameters, LinearAlgebra, Interpolations

@with_kw struct Params
    """ Class with parameters for the incomplete markets model.
    β   : float, agent's discount factor
    ρ   : float, persistence of income process
    σ   : float, st. dev of shocks to income process
    """
    β::Float64 = 1. - 0.08/4
    ρ::Float64 = 0.975
    σ::Float64 = 0.7
end

function geomspace(amin::Float64, amax::Float64, N::Int64; pivot = 0.1)
    """ Constructs logspaced grid for assets - replicates behavior of Python's np.logspace()
    Parameters
    ------------
    amin    : float, lower bound on asset grid
    amax    : float, upper bound on asset grid
    N       : int, number of desired grid points

    Returns
    ------------
    grid    : array(N), grid for assets
    """
    grid = 10 .^ (range(log10(amin+pivot), log10(amax+pivot),length=N)) .- pivot
    grid[1], grid[end] = amin, amax
    return grid
end

function backward_iterate(c₊, a, y, r, r_post, Π, up, up_inv, p::Params)
    """ Implements backward iteration on consumption policy function via EGM and linear interpolation
    Parameters
    ------------
    c₊     : array(N, S), guess for consumption policy tomorrow
    a      : array(N), grid of asset values
    y      : array(S), discretized income process
    r      : float, interest rate that enters Euler equation
    r_post : float, interest rate on today's assets
    Π      : array(S, S), transition matrix for income process
    up     : function, marginal utility of consumption
    up_inv : function, inverse of the first derivative of utility function
    p      : Params, parameters of the model

    Returns
    -----------
    c       : array(N, S), today's consumption policy functions
    a₊      : array(N, S), today's state asset policy functions
    """
    coh = (1 + r_post)*a .+ y'
    c_endog = up_inv( p.β * (1 + r) * up(c₊) * Π')
    c = similar(c_endog)
    for (s, yi) in enumerate(y)
        G = LinearInterpolation(c_endog[:, s] + a, c_endog[:, s],  extrapolation_bc = Line() )
        c[:, s] = G.(coh[:,s])
    end

    # deals with the constraint - could be improved
    a₊ = coh - c
    a₊[a₊ .< a[1]] .= a[1]
    c = coh - a₊

    return c, a₊
end

function ss_policy(a, y, r, Π, up, up_inv, p::Params; maxit = 10000, tol = 1E-9, verbose = true)
    """ Solves for steady state policy function of the incomplete markets model via backward iteration.

    Parameters
    ------------
    a      : array(N), grid of asset values
    y      : array(S), discretized income process
    r      : float, interest rate
    Π      : array(S, S), transition matrix for income process
    up     : function, marginal utility of consumption
    up_inv : function, inverse of the first derivative of utility function
    p      : Params, parameters of the model

    Returns
    -----------
    c       : array(N, S), steady state consumption policy functions
    a₊      : array(N, S), steady state asset policy functions
    """
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


function ss(a, y, r, Π, up, up_inv, p::Params; verbose = true)
    """ Solves for steady state of the incomplete markets model

    Parameters
    ------------
    a      : array(N), grid of asset values
    y      : array(S), discretized income process
    r      : float, interest rate
    Π      : array(S, S), transition matrix for income process
    up     : function, marginal utility of consumption
    up_inv : function, inverse of the first derivative of utility function
    p      : Params, parameters of the model

    Returns
    -----------
    c       : array(N, S), consumption policy functions
    a₊      : array(N, S), asset policy functions
    D       : array(N, S), steady state distribution over assets and income
    C       : float, aggregate consumption
    A       : float, aggregate consumption
    """

    # solve for policy
    c, a₊ = ss_policy(a, y, r, Π, up, up_inv, p, verbose = verbose)

    # solve for distribution
    a₊i = Array{Int64}(undef, length(a), length(y))
    pi_a = Array{Float64}(undef, length(a), length(y))
    for i in 1:length(y)
        a₊i[:, i], pi_a[:, i] = interpolate_policy(a, a₊[:, i])
    end
    D = ergodic_dist(Π, a₊i, pi_a, verbose = verbose);

    #get aggregates
    C = c ⋅ D
    A = a ⋅ sum(D, dims = 2)

    return c, a₊, D, C, A
end


# Partial equilibrium dynamics
function td_PE(a, y, r, Π, up, up_inv, p::Params; rs = nothing, ys = nothing)
    """ Computes simple partial equilibrium transition dynamics for incomplete markets model.

    Parameters
    -----------
    a      : array(N), grid of asset values
    y      : array(S), discretized income process
    r      : float, steady state interest rate
    Π      : array(S, S), transition matrix for income process
    up     : function, marginal utility of consumption
    up_inv : function, inverse of the first derivative of utility function
    p      : Params, parameters of the model

    rs     : array(T), path for interest rate
    ys     : array(T, S), path for income process

    Returns
    -----------
    Path for aggregate consumption and path for aggreate assets
    """
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
