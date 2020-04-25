
function interpolate_policy(x, xq)
    """ Gets discretized policy rule representation of policy function x at query points xq.

    Parameters
    ----------
    x  : array(n), ascending data points
    xq  : array(nq), ascending query points

    Returns
    ---------
    xqi  : array(nq), indices of lower bracketing gridpoints
    pi   : array(nq), weights on lower bracketing gridpoints
    """

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

function forward_iterate(D, Π, k₊i, pi_k)
    """ Update the distribution given discretized policy rule k₊i and weights pi_k.

    Parameters
    -----------
    D    : array (len(k), len(z)), current distribution
    Π    : array, markov transition matrix for exogenous state
    k₊i  : array, indices of discretized policy rule
    pi_k : array, weights on lower gridpoints

    Returns
    -------
    Dnew Π  : array(len(k), len(z)), updated distribution
    """
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

function ergodic_dist(Π, k₊i, pi_k; maxit = 10000, tol = 1E-10)

    """ Computes the ergodic distribution of model with discretized policy rule  k₊i and weights pi_k, given transition matrix for exogenous states Π.

    Parameters
    ---------
    Π    : array, markov transition matrix for exogenous state
    k₊i  : array, indexes of discretized policy rule
    pi_k : array, weights on lower gridpoints

    Returns
    ---------
    D   : array, ergodic distribution of the model
    """

    # start by getting stationary distribution of z
    pr = stationary(Π)
    # assume uniform distribution on k for the initial distribution
    nK = size(k₊i)[1]
    D = pr .* fill(1/nK, nK)
    for it in 1:maxit
        Dnew = forward_iterate(D, Π, k₊i, pi_k)
        if mod(it, 20) ≈ 0 && norm(Dnew - D) < tol
            println("Convergence after $it iterations!")
            break
        end
        D = Dnew
    end
    return D
end
