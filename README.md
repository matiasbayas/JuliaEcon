# JuliaEcon

Computational economics in Julia: heterogeneous-agent models, Chebyshev approximation methods, and Markov chain discretization. Developed during coursework based on Matt Rognlie's computational methods material at Northwestern.

## Structure

```
src/          Julia source modules
notebooks/    Jupyter notebooks with examples and analysis
```

## Models and methods

| File | Description |
|------|-------------|
| `src/markov_approx.jl` | Tauchen and Rouwenhorst methods for discretizing AR(1) processes |
| `src/distribution.jl` | Forward iteration on distributions with discretized policy rules |
| `src/chebyshev.jl` | Chebyshev polynomial interpolation and root-finding utilities |
| `src/neoclassical.jl` | Neoclassical growth model on a discrete grid (EGM) |
| `src/neoclassical_cheb.jl` | Neoclassical growth model with Chebyshev approximation |
| `src/endog_labor.jl` | Neoclassical model with endogenous labor supply |
| `src/incomplete_marketsPE.jl` | Bewley-Huggett-Aiyagari incomplete markets model (partial equilibrium) |
| `src/incomplete_marketsGE.jl` | Incomplete markets with general equilibrium and transition dynamics |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/chebyshev.ipynb` | Chebyshev approximation examples |
| `notebooks/distribution.ipynb` | Distribution computation and visualization |
| `notebooks/incomplete_markets.ipynb` | Incomplete markets model walkthrough |
| `notebooks/MPC.ipynb` | Marginal propensity to consume analysis |
| `notebooks/neoclassical_cheb.ipynb` | Neoclassical model with Chebyshev methods |
| `notebooks/numerical_integration.ipynb` | Numerical integration techniques |
| `notebooks/transitions_PE.ipynb` | Partial equilibrium transition dynamics |
| `notebooks/transitions_GE.ipynb` | General equilibrium transition dynamics |

## Installation

Requires Julia 1.6+. From the repo root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
include("src/incomplete_marketsPE.jl")
using Parameters

# Solve the Bewley-Huggett-Aiyagari model
a, c, a₊, D, C, A = solveIncompleteMarkets(0., 200., 500, Params())
```

## Attribution

Based on material from Matt Rognlie's computational methods course at Northwestern University.

## License

MIT
