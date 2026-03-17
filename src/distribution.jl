# Distribution tools: policy interpolation, forward iteration, and ergodic distribution computation

"""
    interpolate_policy(x, xq)

Discretize a continuous policy function for forward iteration on the distribution.
Returns `(xqi, pi)`: indices of lower bracketing gridpoints and interpolation weights.
"""
function interpolate_policy(x, xq)
    nq, n = size(xq)[1], size(x)[1]
    xqi = Array{Int64}(undef, nq)
    pi = Array{Float64}(undef, nq)

    xi = 1
    xlow = x[1]
    xhigh = x[2]
    for i in 1:nq
        xq_cur = xq[i]
        while xi < n - 1
            if xhigh >= xq_cur
                break
            end
            xi += 1
            xlow = xhigh
            xhigh = x[xi + 1]
        end
        xqi[i] = xi
        pi[i] = (xhigh - xq_cur) / (xhigh - xlow)
    end
    return xqi, pi
end

"""
    forward_iterate(D, Π, k₊i, pi_k)

Update the distribution one period forward given discretized policy rule.
"""
function forward_iterate(D, Π, k₊i, pi_k)
    Dnew = zeros(size(D))
    # first update using endogenous capital policy
    for zi in 1:size(D)[2]
        for ki in 1:size(D)[1]
            i = k₊i[ki, zi]
            pi = pi_k[ki, zi]
            d = D[ki, zi]
            Dnew[i, zi] += d*pi
            Dnew[i+1, zi] += d*(1-pi)
        end
    end
    # now update the exogenous state using Markov matrix Π
    return Dnew * Π
end

"""
    ergodic_dist(Π, k₊i, pi_k)

Compute the ergodic distribution by iterating forward_iterate to convergence.
"""
function ergodic_dist(Π, k₊i, pi_k; maxit = 10000, tol = 1E-10, verbose = true)
    # start by getting stationary distribution of z
    pr = stationary(Π)
    # assume uniform distribution on k for the initial distribution
    nK = size(k₊i)[1]
    D = pr .* fill(1/nK, nK)
    for it in 1:maxit
        Dnew = forward_iterate(D, Π, k₊i, pi_k)
        if mod(it, 20) ≈ 0 && norm(Dnew - D) < tol
            if verbose
                println("Convergence after $it iterations!")
            end
            break
        end
        D = Dnew
    end
    return D
end
