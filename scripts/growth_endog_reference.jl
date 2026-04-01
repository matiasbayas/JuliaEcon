include("../src/endog_labor.jl")
include("../src/distribution.jl")

using JSON
using Parameters
using BasisMatrices
using LinearAlgebra
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

@with_kw struct GrowthReferenceConfig
    preset::String = "legacy"
    beta::Float64 = 0.96
    alpha::Float64 = 0.36
    delta::Float64 = 0.025
    rho::Float64 = 0.9
    sigma_eps::Float64 = 0.01
    sigma::Float64 = 2.0
    nu::Float64 = 0.5
    n_ss_target::Float64 = 1.0 / 3.0
    z_nodes::Int = 9
    shock_method::String = "rouwenhorst"
    tauchen_m::Float64 = 2.0
    cheb_nodes::Int = 15
    k_low_mult::Float64 = 0.5
    k_high_mult::Float64 = 4.0
    n_low_mult::Float64 = 0.2
    n_high_mult::Float64 = 6.0
    maxit::Int = 500
    tol::Float64 = 1.0e-8
    dense_k_points::Int = 400
    output_path::String = "reference/growth_endog_reference.json"
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

function normalize_override_key(key::String)
    return key == "sigma_e" ? "sigma_eps" : key
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
        return GrowthReferenceConfig(
            preset = "legacy",
            beta = 0.95,
            alpha = 0.4,
            delta = 0.05,
            rho = 0.9,
            sigma_eps = 0.2,
            sigma = 1.0,
            nu = 1.0,
            n_ss_target = -1.0,
            z_nodes = 3,
            shock_method = "tauchen",
            tauchen_m = 2.0,
            cheb_nodes = 15,
            k_low_mult = 0.4,
            k_high_mult = 2.5,
            n_low_mult = 0.2,
            n_high_mult = 6.0,
            maxit = 200,
            tol = 5.0e-9,
            dense_k_points = 400,
            output_path = "reference/growth_endog_reference_legacy.json",
        )
    elseif preset == "arde_prompt"
        return GrowthReferenceConfig(
            preset = "arde_prompt",
            cheb_nodes = 21,
            maxit = 700,
            tol = 1.0e-9,
            dense_k_points = 600,
            output_path = "reference/growth_endog_reference_arde_prompt.json",
        )
    else
        error("Unknown preset: $preset")
    end
end

function merge_config(base::GrowthReferenceConfig, overrides::Dict{String, Any})
    allowed = Set(string.(fieldnames(GrowthReferenceConfig)))
    for key in keys(overrides)
        key in allowed || error("Unknown config key: $key")
    end
    return GrowthReferenceConfig(;
        (
            name => get(overrides, string(name), getfield(base, name))
            for name in fieldnames(GrowthReferenceConfig)
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
            overrides[normalize_override_key(string(k))] = v
        end
    end
    preset = get(opts, "preset", get(overrides, "preset", "legacy"))
    cfg = default_config_for_preset(String(preset))

    for (k, v) in opts
        if k in ("preset", "config")
            continue
        end
        overrides[normalize_override_key(k)] = v
    end

    typed = Dict{String, Any}()
    for name in fieldnames(GrowthReferenceConfig)
        key = string(name)
        if !haskey(overrides, key)
            continue
        end
        current = getfield(cfg, name)
        value = overrides[key]
        typed[key] = coerce_override_value(current, value)
    end

    return merge_config(cfg, typed)
end

function to_endog_params(cfg::GrowthReferenceConfig)
    return Params(β = cfg.beta, α = cfg.alpha, δ = cfg.delta, ρ = cfg.rho, σ = cfg.sigma_eps)
end

function utility_marginal(c, cfg::GrowthReferenceConfig)
    if cfg.preset == "legacy"
        return 1.0 ./ c
    end
    return c .^ (-cfg.sigma)
end

function inverse_marginal_utility(mu, cfg::GrowthReferenceConfig)
    if cfg.preset == "legacy"
        return 1.0 ./ mu
    end
    return mu .^ (-1.0 / cfg.sigma)
end

function labor_marginal_utility(n, cfg::GrowthReferenceConfig, chi::Float64)
    if cfg.preset == "legacy"
        return n
    end
    return chi .* n .^ (1.0 / cfg.nu)
end

function production_functions(cfg::GrowthReferenceConfig)
    F(z, k, n) = z .* k .^ cfg.alpha .* n .^ (1.0 - cfg.alpha) .+ (1.0 - cfg.delta) .* k
    Fk(z, k, n) = cfg.alpha .* z .* (n ./ k) .^ (1.0 - cfg.alpha) .+ (1.0 - cfg.delta)
    Fn(z, k, n) = (1.0 - cfg.alpha) .* z .* (k ./ n) .^ cfg.alpha
    return F, Fk, Fn
end

function productivity_process(cfg::GrowthReferenceConfig)
    if cfg.preset == "legacy"
        return ProductivityProcess(to_endog_params(cfg))
    elseif cfg.shock_method == "rouwenhorst"
        return markov_rouwenhorst(cfg.rho, cfg.sigma_eps; N = cfg.z_nodes)
    elseif cfg.shock_method == "tauchen"
        return markov_tauchen(cfg.rho, cfg.sigma_eps; N = cfg.z_nodes, m = cfg.tauchen_m)
    else
        error("Unsupported shock method: $(cfg.shock_method)")
    end
end

function generalized_steady_state(cfg::GrowthReferenceConfig, F)
    capital_labor_ratio = ((1.0 / cfg.beta - 1.0 + cfg.delta) / cfg.alpha) ^ (1.0 / (cfg.alpha - 1.0))
    if cfg.preset == "legacy"
        d1 = (1.0 - cfg.alpha) * capital_labor_ratio ^ cfg.alpha
        d2 = F(1.0, capital_labor_ratio, 1.0) - capital_labor_ratio
        c_ss = sqrt(d1 * d2)
        n_ss = c_ss / d2
        k_ss = capital_labor_ratio * n_ss
        chi = 1.0
    else
        n_ss = cfg.n_ss_target
        k_ss = capital_labor_ratio * n_ss
        y_ss = k_ss ^ cfg.alpha * n_ss ^ (1.0 - cfg.alpha)
        c_ss = y_ss - cfg.delta * k_ss
        chi = utility_marginal(c_ss, cfg) * (1.0 - cfg.alpha) * k_ss ^ cfg.alpha * n_ss ^ (-cfg.alpha - 1.0 / cfg.nu)
    end
    y_ss = k_ss ^ cfg.alpha * n_ss ^ (1.0 - cfg.alpha)
    return (
        c_ss = c_ss,
        n_ss = n_ss,
        k_ss = k_ss,
        y_ss = y_ss,
        z_ss = 1.0,
        chi = chi,
    )
end

function generalized_backward_iterate(k, z, nq_next, cq_next, cfg::GrowthReferenceConfig, chi::Float64, Π, basis)
    F, Fk, Fn = production_functions(cfg)
    nq = similar(nq_next)
    cq = similar(cq_next)
    root_fallbacks = 0
    for (i, z_cur) in enumerate(z)
        pr_z = Π[i, :]
        n_vec = similar(k)
        c_vec = similar(k)
        for (j, k_cur) in enumerate(k)
            h(n_guess) = begin
                mu_c = labor_marginal_utility(n_guess, cfg, chi) / Fn(z_cur, k_cur, n_guess)
                c = inverse_marginal_utility(mu_c, cfg)
                k_next = F(z_cur, k_cur, n_guess) - c
                if k_next < 0.0
                    return -1.0e-6
                end
                c_next = BasisMatrix(basis, Direct(), [k_next]).vals[1] * cq_next
                n_next = max.(BasisMatrix(basis, Direct(), [k_next]).vals[1] * nq_next, eps(Float64))
                expected_rhs = Fk(z', k_next, n_next) .* utility_marginal(c_next, cfg)
                return mu_c .- cfg.beta * (expected_rhs ⋅ pr_z)
            end
            local_nmin = max(cfg.n_low_mult * eps(Float64), eps(Float64))
            local_nmax = cfg.n_high_mult
            fa = h(local_nmin)
            fb = h(local_nmax)
            if isfinite(fa) && isfinite(fb) && signbit(fa) != signbit(fb)
                n_vec[j] = brent(h, local_nmin, local_nmax)
            else
                probe_grid = collect(range(local_nmin, local_nmax, length = 200))
                probe_vals = [h(x) for x in probe_grid]
                bracket_found = false
                for idx in 1:length(probe_grid)-1
                    left_val = probe_vals[idx]
                    right_val = probe_vals[idx + 1]
                    if !isfinite(left_val) || !isfinite(right_val)
                        continue
                    end
                    if signbit(left_val) != signbit(right_val)
                        n_vec[j] = brent(h, probe_grid[idx], probe_grid[idx + 1])
                        bracket_found = true
                        break
                    end
                end
                if !bracket_found
                    best_idx = argmin(abs.(probe_vals))
                    n_vec[j] = probe_grid[best_idx]
                    root_fallbacks += 1
                end
            end
            c_vec[j] = inverse_marginal_utility(
                labor_marginal_utility(n_vec[j], cfg, chi) / Fn(z_cur, k_cur, n_vec[j]),
                cfg,
            )
        end
        nq[:, i] = cheb_interp(n_vec, basis)
        cq[:, i] = cheb_interp(c_vec, basis)
    end
    return nq, cq, root_fallbacks
end

function solve_generalized(cfg::GrowthReferenceConfig)
    F, _, _ = production_functions(cfg)
    steady = generalized_steady_state(cfg, F)
    z, pr, Π = productivity_process(cfg)
    k_nodes, basis = cheb_nodes(cfg.k_low_mult * steady.k_ss, cfg.k_high_mult * steady.k_ss, cfg.cheb_nodes)
    nmin = max(cfg.n_low_mult * steady.n_ss, eps(Float64))
    nmax = cfg.n_high_mult * steady.n_ss

    c0 = repeat((steady.c_ss / steady.k_ss) .* k_nodes', length(z), 1)
    n0 = repeat(steady.n_ss .* z, 1, length(k_nodes))
    cq = cheb_interp(c0', basis)
    nq = cheb_interp(n0', basis)
    total_root_fallbacks = 0

    for it in 1:cfg.maxit
        nq_new, cq_new, root_fallbacks = generalized_backward_iterate(k_nodes, z, nq, cq, cfg, steady.chi, Π, basis)
        total_root_fallbacks += root_fallbacks
        if mod(it, 10) == 0 && norm(nq_new - nq) < cfg.tol
            return (steady = steady, z = z, pr = pr, Π = Π, k = k_nodes, basis = basis, nq = nq_new, cq = cq_new, iterations = it, root_fallbacks = total_root_fallbacks)
        end
        nq = nq_new
        cq = cq_new
    end
    return (steady = steady, z = z, pr = pr, Π = Π, k = k_nodes, basis = basis, nq = nq, cq = cq, iterations = cfg.maxit, root_fallbacks = total_root_fallbacks)
end

function solve_legacy(cfg::GrowthReferenceConfig)
    p = to_endog_params(cfg)
    nq, cq, k, basis = solveNeoclassical(p, cfg.cheb_nodes)
    z, pr, Π = ProductivityProcess(p)
    F, _, _ = production_functions(cfg)
    steady_named = generalized_steady_state(cfg, F)
    return (steady = steady_named, z = z, pr = pr, Π = Π, k = k, basis = basis, nq = nq, cq = cq, iterations = nothing, root_fallbacks = 0)
end

function materialize_solution(sol, cfg::GrowthReferenceConfig)
    F, Fk, Fn = production_functions(cfg)
    c = funeval(sol.cq, sol.basis, sol.k)
    n = funeval(sol.nq, sol.basis, sol.k)
    k_next = F(sol.z', sol.k, n) - c

    kqi = Array{Int64}(undef, length(sol.k), length(sol.z))
    pi_k = Array{Float64}(undef, length(sol.k), length(sol.z))
    for i in 1:length(sol.z)
        kqi[:, i], pi_k[:, i] = interpolate_policy(sol.k, k_next[:, i])
    end
    D = ergodic_dist(sol.Π, kqi, pi_k; verbose = false)

    k_dense = collect(range(sol.k[1], sol.k[end], length = cfg.dense_k_points))
    c_dense = funeval(sol.cq, sol.basis, k_dense)
    n_dense = funeval(sol.nq, sol.basis, k_dense)
    k_next_dense = F(sol.z', k_dense, n_dense) - c_dense

    euler_errors = zeros(length(k_dense), length(sol.z))
    labor_errors = zeros(length(k_dense), length(sol.z))
    for (i, z_cur) in enumerate(sol.z)
        k_next_col = k_next_dense[:, i]
        c_next = funeval(sol.cq, sol.basis, k_next_col)
        n_next = max.(funeval(sol.nq, sol.basis, k_next_col), eps(Float64))
        rhs = vec(cfg.beta .* (sol.Π[i, :]' * (Fk(sol.z', k_next_col, n_next) .* utility_marginal(c_next, cfg))'))
        c_implied_dynamic = inverse_marginal_utility(rhs, cfg)
        euler_errors[:, i] = c_implied_dynamic ./ c_dense[:, i] .- 1.0

        c_implied_static = inverse_marginal_utility(
            labor_marginal_utility(n_dense[:, i], cfg, sol.steady.chi) ./ Fn(z_cur, k_dense, n_dense[:, i]),
            cfg,
        )
        labor_errors[:, i] = c_implied_static ./ c_dense[:, i] .- 1.0
    end

    k_weights = vec(sum(D, dims = 2))
    z_weights = vec(sum(D, dims = 1))

    moments = Dict(
        "ergodic" => Dict(
            "k_mean" => sum(sol.k .* k_weights),
            "c_mean" => sum(c .* D),
            "n_mean" => sum(n .* D),
            "y_mean" => sum((sol.z' .* sol.k .^ cfg.alpha .* n .^ (1.0 - cfg.alpha)) .* D),
            "mass_sum" => sum(D),
        ),
    )

    diagnostics = Dict(
        "schema_version" => "1.0",
        "max_abs_euler" => maximum(abs.(euler_errors)),
        "max_abs_labor" => maximum(abs.(labor_errors)),
        "error_metric" => "consumption_equivalent_relative_error",
        "policy_shape" => Dict(
            "k_next_monotonic_violations" => sum(count(diff(k_next[:, i]) .< -1.0e-10) for i in 1:size(k_next, 2)),
        ),
        "ergodic_distribution" => Dict(
            "mass_sum" => sum(D),
            "min_mass" => minimum(D),
            "max_mass" => maximum(D),
        ),
        "iterations" => sol.iterations,
        "root_fallbacks" => sol.root_fallbacks,
        "nan_or_inf_detected" => any(x -> !isfinite(x), Iterators.flatten((c_dense, n_dense, k_next_dense, euler_errors, labor_errors))),
    )

    # Export 2D arrays in explicit row-major form so downstream scorers can
    # interpret them consistently as [k_index][z_index].
    matrix_rows(A) = [collect(A[i, :]) for i in axes(A, 1)]

    return Dict(
        "schema_version" => "1.0",
        "problem_id" => "growth_endogenous_labor",
        "reference_source" => Dict(
            "repo" => "JuliaEcon",
            "preset" => cfg.preset,
            "script" => "scripts/growth_endog_reference.jl",
        ),
        "params" => Dict(
            "beta" => cfg.beta,
            "alpha" => cfg.alpha,
            "delta" => cfg.delta,
            "rho" => cfg.rho,
            "sigma_e" => cfg.sigma_eps,
            "sigma" => cfg.sigma,
            "nu" => cfg.nu,
            "chi" => sol.steady.chi,
            "n_ss_target" => cfg.n_ss_target < 0 ? nothing : cfg.n_ss_target,
        ),
        "steady_state" => Dict(
            "c_ss" => sol.steady.c_ss,
            "n_ss" => sol.steady.n_ss,
            "k_ss" => sol.steady.k_ss,
            "y_ss" => sol.steady.y_ss,
            "z_ss" => sol.steady.z_ss,
            "chi" => sol.steady.chi,
        ),
        "grids" => Dict(
            "k_grid" => collect(sol.k),
            "z_grid" => collect(sol.z),
            "k_dense_grid" => k_dense,
        ),
        "array_layout" => Dict(
            "state_axes" => ["k_index", "z_index"],
            "transition_axes" => ["z_current", "z_next"],
        ),
        "transition" => Dict(
            "Pz" => matrix_rows(sol.Π),
            "pi_z" => vec(sol.pr),
        ),
        "approximation" => Dict(
            "basis_type" => "chebyshev",
            "consumption_coefficients" => matrix_rows(sol.cq),
            "labor_coefficients" => matrix_rows(sol.nq),
        ),
        "decision_rule" => Dict(
            "consumption" => matrix_rows(c),
            "labor" => matrix_rows(n),
            "capital_next" => matrix_rows(k_next),
            "consumption_dense" => matrix_rows(c_dense),
            "labor_dense" => matrix_rows(n_dense),
            "capital_next_dense" => matrix_rows(k_next_dense),
        ),
        "distribution" => Dict(
            "ergodic_state" => matrix_rows(D),
            "ergodic_marginal_k" => k_weights,
            "ergodic_marginal_z" => z_weights,
        ),
        "moments" => moments,
        "diagnostics" => diagnostics,
    )
end

function solve_reference(cfg::GrowthReferenceConfig)
    sol = cfg.preset == "legacy" ? solve_legacy(cfg) : solve_generalized(cfg)
    return materialize_solution(sol, cfg)
end

function write_reference(output_path::String, reference_data)
    resolved_output = isabspath(output_path) ? output_path : normpath(joinpath(REPO_ROOT, output_path))
    mkpath(dirname(resolved_output))
    open(resolved_output, "w") do io
        JSON.print(io, reference_data, 2)
    end
    return resolved_output
end

function main(args)
    cfg = load_config(args)
    reference_data = solve_reference(cfg)
    resolved_output = write_reference(cfg.output_path, reference_data)
    println("wrote reference artifact to $(resolved_output)")
    println(@sprintf("preset=%s max_abs_euler=%.6e max_abs_labor=%.6e", cfg.preset, reference_data["diagnostics"]["max_abs_euler"], reference_data["diagnostics"]["max_abs_labor"]))
end

main(ARGS)
