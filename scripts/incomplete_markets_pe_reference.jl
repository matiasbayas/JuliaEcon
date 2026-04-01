include("../src/incomplete_marketsPE.jl")

using JSON
using Parameters
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

@with_kw struct PEReferenceConfig
    preset::String = "legacy"
    beta::Float64 = 1.0 - 0.08 / 4.0
    gamma::Float64 = 1.0
    rho::Float64 = 0.975
    sigma_y::Float64 = 0.7
    sigma_mode::String = "stationary"
    r::Float64 = 0.01 / 4.0
    amin::Float64 = 0.0
    amax::Float64 = 200.0
    asset_nodes::Int = 500
    shock_nodes::Int = 7
    shock_method::String = "rouwenhorst"
    maxit::Int = 10_000
    tol::Float64 = 1.0e-9
    dist_maxit::Int = 10_000
    dist_tol::Float64 = 1.0e-10
    output_path::String = "reference/incomplete_markets_pe_reference_legacy.json"
end

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if i == length(args) || startswith(args[i + 1], "--")
                opts[key] = "true"
                i += 1
            else
                opts[key] = args[i + 1]
                i += 2
            end
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return opts
end

function coerce_override_value(current, value)
    if current isa Int
        return value isa String ? parse(Int, value) : Int(value)
    elseif current isa Float64
        return value isa String ? parse(Float64, value) : Float64(value)
    elseif current isa Bool
        return value isa String ? lowercase(value) == "true" : Bool(value)
    elseif current isa String
        return String(value)
    end
    return value
end

function default_config_for_preset(preset::String)
    if preset == "legacy"
        return PEReferenceConfig(
            preset = "legacy",
            gamma = 1.0,
            sigma_mode = "stationary",
            output_path = "reference/incomplete_markets_pe_reference_legacy.json",
        )
    elseif preset == "arde_prompt"
        return PEReferenceConfig(
            preset = "arde_prompt",
            gamma = 2.0,
            sigma_mode = "innovation",
            output_path = "reference/incomplete_markets_pe_reference_arde_prompt.json",
        )
    else
        error("Unknown preset: $preset")
    end
end

function merge_config(base::PEReferenceConfig, overrides::Dict{String, Any})
    allowed = Set(string.(fieldnames(PEReferenceConfig)))
    for key in keys(overrides)
        key in allowed || error("Unknown config key: $key")
    end
    return PEReferenceConfig(;
        (
            name => get(overrides, string(name), getfield(base, name))
            for name in fieldnames(PEReferenceConfig)
        )...
    )
end

function load_config(args)
    opts = parse_args(args)
    if haskey(opts, "output") && !haskey(opts, "output_path")
        opts["output_path"] = opts["output"]
    end
    overrides = Dict{String, Any}()
    if haskey(opts, "config")
        file_overrides = JSON.parsefile(opts["config"])
        for (k, v) in file_overrides
            overrides[string(k)] = v
        end
    end
    preset = get(opts, "preset", get(overrides, "preset", "legacy"))
    cfg = default_config_for_preset(String(preset))

    for (k, v) in opts
        if k in ("preset", "config")
            continue
        end
        overrides[k] = v
    end

    typed = Dict{String, Any}()
    for name in fieldnames(PEReferenceConfig)
        key = string(name)
        if !haskey(overrides, key)
            continue
        end
        current = getfield(cfg, name)
        typed[key] = coerce_override_value(current, overrides[key])
    end
    return merge_config(cfg, typed)
end

function utility_marginal(c::AbstractArray, gamma::Float64)
    if isapprox(gamma, 1.0; atol = 1.0e-12)
        return 1.0 ./ c
    end
    return c .^ (-gamma)
end

function inverse_marginal_utility(mu::AbstractArray, gamma::Float64)
    if isapprox(gamma, 1.0; atol = 1.0e-12)
        return 1.0 ./ mu
    end
    return mu .^ (-1.0 / gamma)
end

function stationary_sigma(cfg::PEReferenceConfig)
    if cfg.sigma_mode == "stationary"
        return cfg.sigma_y
    elseif cfg.sigma_mode == "innovation"
        return cfg.sigma_y / sqrt(1.0 - cfg.rho ^ 2)
    end
    error("Unsupported sigma_mode: $(cfg.sigma_mode)")
end

function income_process(cfg::PEReferenceConfig)
    σ = stationary_sigma(cfg)
    if cfg.shock_method == "rouwenhorst"
        y, _, Π = markov_rouwenhorst(cfg.rho, σ; N = cfg.shock_nodes)
    elseif cfg.shock_method == "tauchen"
        y, _, Π = markov_tauchen(cfg.rho, σ; N = cfg.shock_nodes)
    else
        error("Unsupported shock_method: $(cfg.shock_method)")
    end
    return y, Π
end

function backward_iterate_general(c_next, a, y, r, Π, cfg::PEReferenceConfig)
    coh = (1 + r) * a .+ y'
    mu_next = utility_marginal(c_next, cfg.gamma)
    c_endog = inverse_marginal_utility(cfg.beta * (1 + r) * (mu_next * Π'), cfg.gamma)
    c = similar(c_endog)
    for s in eachindex(y)
        G = LinearInterpolation(c_endog[:, s] + a, c_endog[:, s], extrapolation_bc = Line())
        c[:, s] = G.(coh[:, s])
    end

    a_next = coh - c
    a_next[a_next .< a[1]] .= a[1]
    c = coh - a_next
    return c, a_next
end

function solve_policies(cfg::PEReferenceConfig)
    @assert cfg.beta * (1 + cfg.r) <= 1 "β(1+r) > 1: problem is not well-defined"
    y, Π = income_process(cfg)
    a = geomspace(cfg.amin, cfg.amax, cfg.asset_nodes)
    c = 0.2 * ((1 + cfg.r) * a .+ y')

    iterations = cfg.maxit
    converged = false
    diff = Inf
    a_next = zeros(size(c))
    for it in 1:cfg.maxit
        c_new, a_new = backward_iterate_general(c, a, y, cfg.r, Π, cfg)
        diff = norm(c_new - c)
        c = c_new
        a_next = a_new
        if mod(it, 10) == 0 && diff < cfg.tol
            iterations = it
            converged = true
            break
        end
    end
    if !converged
        iterations = cfg.maxit
    end
    return a, y, Π, c, a_next, converged, iterations, diff
end

function stationary_distribution(a, y, Π, a_next, cfg::PEReferenceConfig)
    a_next_i = Array{Int64}(undef, length(a), length(y))
    pi_a = Array{Float64}(undef, length(a), length(y))
    for i in 1:length(y)
        a_next_i[:, i], pi_a[:, i] = interpolate_policy(a, a_next[:, i])
    end
    D = ergodic_dist(Π, a_next_i, pi_a; maxit = cfg.dist_maxit, tol = cfg.dist_tol, verbose = false)
    return D, a_next_i, pi_a
end

function aggregates(a, c, a_next, D)
    C = c ⋅ D
    A = a ⋅ sum(D, dims = 2)
    constraint_mass = sum(D[1, :])
    return (
        aggregate_consumption = Float64(C),
        aggregate_assets = Float64(A),
        borrowing_constraint_mass = Float64(constraint_mass),
    )
end

function monotone_non_decreasing(x::AbstractVector)
    return all(diff(x) .>= -1.0e-10)
end

function build_payload(cfg::PEReferenceConfig, a, y, Π, c, a_next, D, converged, iterations, final_diff)
    aggs = aggregates(a, c, a_next, D)
    payload = Dict(
        "problem_id" => "incomplete_markets_pe",
        "reference_source" => "JuliaEcon/scripts/incomplete_markets_pe_reference.jl",
        "preset" => cfg.preset,
        "parameters" => Dict(
            "beta" => cfg.beta,
            "gamma" => cfg.gamma,
            "rho" => cfg.rho,
            "sigma_y" => cfg.sigma_y,
            "sigma_mode" => cfg.sigma_mode,
            "r" => cfg.r,
            "amin" => cfg.amin,
            "amax" => cfg.amax,
            "asset_nodes" => cfg.asset_nodes,
            "shock_nodes" => cfg.shock_nodes,
            "shock_method" => cfg.shock_method,
        ),
        "grids" => Dict(
            "asset_grid" => collect(a),
            "income_grid" => collect(y),
        ),
        "transition" => Dict(
            "income_transition" => Π,
        ),
        "decision_rule" => Dict(
            "consumption" => c,
            "assets_next" => a_next,
        ),
        "distribution" => Dict(
            "stationary" => D,
        ),
        "aggregates" => Dict(
            "assets" => aggs.aggregate_assets,
            "consumption" => aggs.aggregate_consumption,
            "borrowing_constraint_mass" => aggs.borrowing_constraint_mass,
        ),
        "diagnostics" => Dict(
            "solver_family" => "EGM_backward_iteration",
            "converged" => converged,
            "iterations" => iterations,
            "final_policy_diff" => final_diff,
            "distribution_mass" => sum(D),
            "policy_monotone_assets_next" => [monotone_non_decreasing(vec(a_next[:, i])) for i in 1:size(a_next, 2)],
            "policy_monotone_consumption" => [monotone_non_decreasing(vec(c[:, i])) for i in 1:size(c, 2)],
        ),
    )
    return payload
end

function write_json(path::String, obj)
    full_path = isabspath(path) ? path : joinpath(REPO_ROOT, path)
    mkpath(dirname(full_path))
    open(full_path, "w") do io
        JSON.print(io, obj, 2)
    end
    return full_path
end

function main(args)
    cfg = load_config(args)
    a, y, Π, c, a_next, converged, iterations, final_diff = solve_policies(cfg)
    D, _, _ = stationary_distribution(a, y, Π, a_next, cfg)
    payload = build_payload(cfg, a, y, Π, c, a_next, D, converged, iterations, final_diff)
    full_path = write_json(cfg.output_path, payload)
    @printf("Wrote %s\n", full_path)
    @printf("converged=%s iterations=%d final_diff=%.6e\n", string(converged), iterations, final_diff)
    @printf("A=%.10f C=%.10f mass_bc=%.10f\n",
        payload["aggregates"]["assets"],
        payload["aggregates"]["consumption"],
        payload["aggregates"]["borrowing_constraint_mass"],
    )
end

main(ARGS)
