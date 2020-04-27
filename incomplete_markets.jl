include("markov_approx.jl")
include("distribution.jl")
using Parameters, LinearAlgebra, Interpolations

@with_kw struct Params
    β::Float64 = 1. - 0.08/4
    ρ::Float64 = 0.975
    σ::Float64 = 0.7
end

function geomspace(amin::Float64, amax::Float64, N::Int64; pivot = 0.1)
    grid = 10 .^ (range(log10(amin+pivot), log10(amax+pivot),length=N)) .- pivot
    grid[1], grid[end] = amin, amax
    return grid
end

function IncomeProcess(p::Params)
    y, p, Π = rouwenhorst(p.ρ, p.σ, N=7)
end

function backward_iterate(c₊, a, y, r, r_post, Π, up, up_inv, p::Params)

    coh = (1+r_post)*a .+ y'
    c_endog = up_inv( p.β * (1+r) * up(c₊) * Π')
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

function ss_policy(a, y, r, Π, up, up_inv, p::Params; maxit = 10000, tol = 1E-9)

    # initial guess for policy function
    c = 0.2 * ((1+r)*a .+ y')

    for it in 1:maxit
        c_new, a_new = backward_iterate(c, a, y, r, r, Π, up, up_inv, p)
        if mod(it, 10) ≈ 0 && norm(c_new - c) < tol
            #println("Convergence in $it iterations!")
            return c_new, a_new
        end
        c = c_new
    end
end

function solveIncompleteMarkets(amin::Float64, amax::Float64, N::Int64, p::Params)

    r = 0.01/4
    @assert p.β * (1 + r) ≤ 1 # need this for problem to be well defined
    y, pr, Π = IncomeProcess(p)
    up(c) = 1 ./ c
    up_inv(c) = 1 ./ c
    a = geomspace(amin, amax, N)

    # solve for policy
    c, a₊ = ss_policy(a, y, r, Π, up, up_inv, p)

    # solve for distribution
    a₊i = Array{Int64}(undef, length(a), length(y))
    pi_a = Array{Float64}(undef, length(a), length(y))
    for i in 1:length(y)
        a₊i[:, i], pi_a[:, i] = interpolate_policy(a, a₊[:, i])
    end
    D = ergodic_dist(Π, a₊i, pi_a; maxit = 10000, tol = 1E-10);

    #get aggregates
    C = c ⋅ D
    A = a ⋅ sum(D, dims = 2)

    return a, c, a₊, D, C, A
end

#a, c, a₊, D, C, A = solveIncompleteMarkets(0., 200., 500, Params());

#using BenchmarkTools
#@btime solveIncompleteMarkets_ss(0., 200., 500, Params())
