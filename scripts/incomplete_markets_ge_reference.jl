include("../src/incomplete_marketsGE.jl")

using JSON
using Parameters
using Printf
using QuantEcon

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

@with_kw struct GEReferenceConfig
    preset::String = "pdr_table_calibration"
    beta::Float64 = 0.988
    gamma::Float64 = 1.0
    rho::Float64 = 0.966
    sigma_y::Float64 = 0.7
    sigma_mode::String = "stationary"
    bond_supply::Float64 = 5.0
    amin::Float64 = 0.0
    amax::Float64 = 250.0
    asset_nodes::Int = 500
    shock_nodes::Int = 8
    shock_method::String = "rouwenhorst"
    root_lower::Float64 = -0.02
    root_upper_slack::Float64 = 2.0e-3
    after_tax_upper_buffer::Float64 = 1.0e-3
    output_path::String = "reference/incomplete_markets_ge_reference_pdr_table_calibration.json"
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
    if preset == "pdr_table_calibration"
        return GEReferenceConfig(
            preset = "pdr_table_calibration",
            beta = 0.988,
            gamma = 1.0,
            rho = 0.966,
            sigma_y = 0.7,
            sigma_mode = "stationary",
            bond_supply = 5.0,
            amin = 0.0,
            amax = 250.0,
            asset_nodes = 500,
            shock_nodes = 8,
            shock_method = "rouwenhorst",
            root_lower = -0.02,
            output_path = "reference/incomplete_markets_ge_reference_pdr_table_calibration.json",
        )
    end
    error("Unknown preset: $preset")
end

function merge_config(base::GEReferenceConfig, overrides::Dict{String, Any})
    allowed = Set(string.(fieldnames(GEReferenceConfig)))
    for key in keys(overrides)
        key in allowed || error("Unknown config key: $key")
    end
    return GEReferenceConfig(;
        (
            name => get(overrides, string(name), getfield(base, name))
            for name in fieldnames(GEReferenceConfig)
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
    preset = get(opts, "preset", get(overrides, "preset", "pdr_table_calibration"))
    cfg = default_config_for_preset(String(preset))

    for (k, v) in opts
        if k in ("preset", "config")
            continue
        end
        overrides[k] = v
    end

    typed = Dict{String, Any}()
    for name in fieldnames(GEReferenceConfig)
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

function stationary_sigma(cfg::GEReferenceConfig)
    if cfg.sigma_mode == "stationary"
        return cfg.sigma_y
    elseif cfg.sigma_mode == "innovation"
        return cfg.sigma_y / sqrt(1.0 - cfg.rho ^ 2)
    end
    error("Unsupported sigma_mode: $(cfg.sigma_mode)")
end

function income_process(cfg::GEReferenceConfig)
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

function monotone_non_decreasing(x::AbstractVector)
    return all(diff(x) .>= -1.0e-10)
end

function euler_error_pdr_style(a, Π, c, a_next, r, cfg::GEReferenceConfig)
    c_right = similar(c)
    for i in 1:size(c, 2)
        c_plus = similar(c)
        for j in 1:size(c, 2)
            c_pol = LinearInterpolation(a, vec(c[:, j]), extrapolation_bc = Line())
            c_plus[:, j] = c_pol.(a_next[:, i])
        end
        c_right[:, i] = inverse_marginal_utility(
            cfg.beta * (1 + r) * (utility_marginal(c_plus, cfg.gamma) * Π[i, :]),
            cfg.gamma,
        )
    end
    err = c_right ./ c .- 1
    err[a_next .<= cfg.amin] .= 0.0
    return Dict(
        "euler_error_style" => "pdr_linear_interpolation",
        "max_abs_euler_error" => maximum(abs.(err)),
        "mean_abs_euler_error" => mean(abs.(err)),
        "max_euler_error" => maximum(err),
        "min_euler_error" => minimum(err),
        "constrained_state_count" => count(a_next .<= cfg.amin),
        "interior_state_count" => count(a_next .> cfg.amin),
    )
end

function root_upper_bound(cfg::GEReferenceConfig)
    impatience_bound = 1.0 / cfg.beta - 1.0 - cfg.root_upper_slack
    after_tax_bound = (1.0 - cfg.after_tax_upper_buffer) / cfg.bond_supply
    upper = min(impatience_bound, after_tax_bound)
    upper > cfg.root_lower || error("No admissible root bracket: root_upper <= root_lower")
    return upper
end

function solve_equilibrium(cfg::GEReferenceConfig)
    @assert cfg.beta > 0.0 "beta must be positive"
    @assert cfg.gamma > 0.0 "gamma must be positive"
    @assert cfg.bond_supply > 0.0 "bond_supply must be positive"
    @assert cfg.root_lower > -1.0 "root_lower must satisfy 1 + r > 0"

    y, Π = income_process(cfg)
    a = geomspace(cfg.amin, cfg.amax, cfg.asset_nodes)
    p = Params(β = cfg.beta, ρ = cfg.rho, σ = stationary_sigma(cfg))
    up(c) = utility_marginal(c, cfg.gamma)
    up_inv(mu) = inverse_marginal_utility(mu, cfg.gamma)

    function asset_market_excess(r)
        disposable_income_factor = 1.0 - r * cfg.bond_supply
        net_income = disposable_income_factor .* y
        _, _, _, _, A = ss(a, net_income, r, Π, up, up_inv, p; verbose = false)
        return A - cfg.bond_supply
    end

    r_lo = cfg.root_lower
    r_hi = root_upper_bound(cfg)
    excess_lo = asset_market_excess(r_lo)
    excess_hi = asset_market_excess(r_hi)
    excess_lo * excess_hi < 0.0 || error(
        @sprintf(
            "No sign change in asset market excess: excess(%.6f)=%.6f excess(%.6f)=%.6f",
            r_lo, excess_lo, r_hi, excess_hi,
        ),
    )

    r_star = brent(asset_market_excess, r_lo, r_hi)
    disposable_income_factor = 1.0 - r_star * cfg.bond_supply
    net_income = disposable_income_factor .* y
    c, a_next, D, C, A = ss(a, net_income, r_star, Π, up, up_inv, p; verbose = false)
    market_error = A - cfg.bond_supply
    return (
        asset_grid = a,
        income_grid = y,
        income_transition = Π,
        net_income_grid = net_income,
        interest_rate = r_star,
        disposable_income_factor = disposable_income_factor,
        consumption = c,
        assets_next = a_next,
        stationary_distribution = D,
        aggregate_consumption = Float64(C),
        aggregate_assets = Float64(A),
        market_clearing_error = Float64(market_error),
        borrowing_constraint_mass = Float64(sum(D[1, :])),
        bracket = (lower = r_lo, upper = r_hi, excess_lower = excess_lo, excess_upper = excess_hi),
    )
end

function build_payload(cfg::GEReferenceConfig, sol)
    euler_diag = euler_error_pdr_style(
        sol.asset_grid,
        sol.income_transition,
        sol.consumption,
        sol.assets_next,
        sol.interest_rate,
        cfg,
    )
    return Dict(
        "problem_id" => "incomplete_markets_ge",
        "reference_source" => "JuliaEcon/scripts/incomplete_markets_ge_reference.jl",
        "preset" => cfg.preset,
        "parameters" => Dict(
            "beta" => cfg.beta,
            "gamma" => cfg.gamma,
            "rho" => cfg.rho,
            "sigma_y" => cfg.sigma_y,
            "sigma_mode" => cfg.sigma_mode,
            "bond_supply" => cfg.bond_supply,
            "amin" => cfg.amin,
            "amax" => cfg.amax,
            "asset_nodes" => cfg.asset_nodes,
            "shock_nodes" => cfg.shock_nodes,
            "shock_method" => cfg.shock_method,
        ),
        "closure" => Dict(
            "balanced_budget_government" => true,
            "household_disposable_income" => "(1 - r * B) * y",
            "implementation_note" => "The reference implementation folds the balanced-budget tax/subsidy system into household disposable income and solves the implied one-dimensional market-clearing problem in r.",
        ),
        "equilibrium" => Dict(
            "interest_rate" => sol.interest_rate,
            "disposable_income_factor" => sol.disposable_income_factor,
            "market_clearing_error" => sol.market_clearing_error,
            "root_bracket" => Dict(
                "lower" => sol.bracket.lower,
                "upper" => sol.bracket.upper,
                "excess_lower" => sol.bracket.excess_lower,
                "excess_upper" => sol.bracket.excess_upper,
            ),
        ),
        "grids" => Dict(
            "asset_grid" => collect(sol.asset_grid),
            "income_grid" => collect(sol.income_grid),
            "net_income_grid" => collect(sol.net_income_grid),
        ),
        "transition" => Dict(
            "income_transition" => sol.income_transition,
        ),
        "decision_rule" => Dict(
            "consumption" => sol.consumption,
            "assets_next" => sol.assets_next,
        ),
        "distribution" => Dict(
            "stationary" => sol.stationary_distribution,
        ),
        "aggregates" => Dict(
            "assets" => sol.aggregate_assets,
            "consumption" => sol.aggregate_consumption,
            "borrowing_constraint_mass" => sol.borrowing_constraint_mass,
        ),
        "diagnostics" => Dict(
            "solver_family" => "EGM_backward_iteration_plus_brent_market_clearing",
            "distribution_mass" => sum(sol.stationary_distribution),
            "policy_monotone_assets_next" => [monotone_non_decreasing(vec(sol.assets_next[:, i])) for i in 1:size(sol.assets_next, 2)],
            "policy_monotone_consumption" => [monotone_non_decreasing(vec(sol.consumption[:, i])) for i in 1:size(sol.consumption, 2)],
            "euler_error" => euler_diag,
        ),
    )
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
    sol = solve_equilibrium(cfg)
    payload = build_payload(cfg, sol)
    full_path = write_json(cfg.output_path, payload)
    @printf("Wrote %s\n", full_path)
    @printf("r*=%.12f disposable_income_factor=%.12f market_error=%.6e\n",
        payload["equilibrium"]["interest_rate"],
        payload["equilibrium"]["disposable_income_factor"],
        payload["equilibrium"]["market_clearing_error"],
    )
    @printf("A=%.10f C=%.10f mass_bc=%.10f\n",
        payload["aggregates"]["assets"],
        payload["aggregates"]["consumption"],
        payload["aggregates"]["borrowing_constraint_mass"],
    )
end

main(ARGS)
