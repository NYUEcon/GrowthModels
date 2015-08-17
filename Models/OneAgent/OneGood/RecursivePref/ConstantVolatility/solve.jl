using CompEcon: qnwnorm
using Dierckx


function envelope_method(nx=15, nk=35, nϵ1=5;maxiter=500, tol=1e-4)

    # Useful parameters
    nshocks = nϵ1
    dist = 10.
    iter = 1

    # Model Parameters
    A, B, vbar, lgbar = 0., 1., .015.^2, .004
    ρ, α, β = -1., -9., .99
    η, ν, δ = .33, .25, .025
    νinv, ρinv, αinv = 1/ν, 1/ρ, 1/α

    # Create some basic functions
    _Y(k) = (η .* k.^ν + (1 - η)).^(νinv)


    # Initialize the arrays
    ϵ, Π = qnwnorm(nϵ1, 0., 1.)
    xgrid = linspace(-.05, .05, nx)
    kgrid = linspace(0., 55., nk)
    xpgrid = Array(Float64, nshocks, nx)
    gpgrid = Array(Float64, nshocks, nx)
    for i=1:nx
        _x = xgrid[i]
        for j=1:nshocks
            xpgrid[j, i] = A * _x + B * sqrt(vbar) * ϵ[j]
            gpgrid[j, i] = exp(lgbar + xpgrid[j, i])
        end

    end

    # Initialize Value Function and Interpolant
    JT = (((1 - β) * (_Y(kgrid) + (1 - δ)*kgrid).^(ρ)).^(ρinv)) ./ 500
    Jvals = repeat(JT, inner=[1, nx])
    Jupd = copy(Jvals)
    Jspl = Spline2D(kgrid, xgrid, Jvals; kx=1, ky=1, s=0.0)
    χpolicy = Array(Float64, nk, nx)

    # Solve this bad boy
    while dist>tol && iter<maxiter

        iter += 1
        # Given the current vf, find the χ that solves the env condition
        for xind=1:nx
            # Pull out current x
            xt = xgrid[xind]

            # Create a 1 dimensional interpolant so I can take derivative
            Jvalsi = Jvals[:, xind]
            Jkspl = Spline1D(kgrid, Jvalsi, k=1, bc="nearest", s=0.0)

            for kind=1:nk
                # Pull out current k
                kt = kgrid[kind]

                # Get the pieces that we care about
                yt = _Y(kt)
                dJ_k = derivative(Jkspl, kt)
                dJ_k = dJ_k < 1e-12 ? 0. : dJ_k
                Jt = evaluate(Jspl, kt, xt)
                ct = (dJ_k/((1-β)*Jt^(1-ρ)*(yt^(1-ν)*η*kt^(ν-1)+1-δ)))^(1./(ρ-1))
                χt = yt + (1 - δ)*kt - ct
                χpolicy[kind, xind] = χt
                expterm = 0.

                for shockind=1:nshocks
                    xtp1 = xpgrid[shockind, xind]
                    gtp1 = gpgrid[shockind, xind]
                    jtp1 = evaluate(Jspl, χt/gtp1, xtp1)
                    expterm += (jtp1 * gtp1)^α
                end
                μ = expterm^αinv

                Jupd[kind, xind] = ((1 - β)*ct^ρ + β*μ^ρ)^(ρinv)

            end
        end

        dist = maxabs(Jvals - Jupd)
        copy!(Jvals, Jupd)
        Jspl = Spline2D(kgrid, xgrid, Jvals)
        @show iter, dist

    end

    return Jspl
end