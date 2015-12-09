module BFZ

using CompEcon: qnwnorm
using Interpolations

type BFZModel
    A::Float64
    B::Float64
    vbar::Float64
    lgbar::Float64
    ρ::Float64
    α::Float64
    β::Float64
    η::Float64
    ν::Float64
    δ::Float64
    ϵ::Vector{Float64}
    Π::Vector{Float64}
    x::AbstractVector
    k::AbstractVector
    xp::Matrix{Float64}
    gp::Matrix{Float64}
end

function BFZModel(;ρ=-1.0, α=-9.0, β=0.99, η=1/3, ν=0.1, δ=0.025, lgbar=0.004,
                   A=0.0, B=1.0, vbar=0.015^2, nk=35, nx=15, nϵ1=5)

    # quadrature weights and nodes
    ϵ, Π = qnwnorm(nϵ1, 0.0, 1.0)

    # grids for x and k
    x = linspace(-0.05, 0.05, nx)
    k = linspace(15.0, 30.0, nk)

    # grids for x_{t+1} and g_{t+1} -- apply law of motion
    xp = Array(Float64, nϵ1, nx)
    gp = Array(Float64, nϵ1, nx)

    for i=1:nx
        _x = x[i]
        for j=1:nϵ1
            xp[j, i] = A*_x + B*sqrt(vbar)*ϵ[j]
            gp[j, i] = exp(lgbar + xp[j, i])
        end

    end

    BFZModel(A, B, vbar, lgbar, ρ, α, β, η, ν, δ, ϵ, Π, x, k, xp, gp)
end

## Helpful functions
_Y(m::BFZModel, k) = (m.η .* k.^m.ν + (1 - m.η)).^(1/m.ν)

"""
Initializes the guess of the value function on the grid to be a small fraction
of consuming production each period. Use the same value for all x
"""
function initialize(m::BFZModel)
    JT = (((1-m.β) * (_Y(m, m.k) + (1-m.δ)*m.k).^(m.ρ)).^(1/m.ρ)) ./ 500
    Jvals = repeat(JT, inner=[1, length(m.x)])
    itp = scale(interpolate(Jvals, BSpline(Quadratic(Line())), OnGrid()),
                m.k, m.x)
    Jvals, itp
end

function envelope_method(m::BFZModel;maxiter=500, tol=1e-4)
    # simplify notation
    β, δ, ρ, η, ν, α = m.β, m.δ, m.ρ, m.η, m.ν, m.α
    nshocks = length(m.ϵ)
    nk, nx = length(m.k), length(m.x)

    # initialize interpolant for value function
    Jvals, itp = initialize(m)
    χpolicy = Array(Float64, nk, nx)
    Jupd = similar(χpolicy)

    # intermediate helper variables
    err = 100.0
    iter = 0
    start_time = time()

    # Solve this bad boy
    while err>tol && iter<maxiter

        iter += 1
        # Given the current vf, find the χ that solves the env condition
        for i_x=1:nx
            # Pull out current x
            xt = m.x[i_x]

            for i_k=1:nk
                # Pull out current k
                kt = m.k[i_k]

                yt = _Y(m, kt)  # production

                dJ_k = gradient(itp, kt, xt)[1]  # evaluate dJ/dk
                dJ_k = dJ_k < 1e-12 ? 0. : dJ_k

                Jt = itp[kt, xt] # evaluate J

                # use envelope and budget constraint to get c and χ, then save χ
                # in policy matrix
                ct = (dJ_k/((1-β)*Jt^(1-ρ)*(yt^(1-ν)*η*kt^(ν-1)+1-δ)))^(1./(ρ-1))
                χt = yt + (1 - δ)*kt - ct
                χpolicy[i_k, i_x] = χt

                # calculate the certainty equivalent
                expterm = 0.
                for i_eps=1:nshocks
                    xp = m.xp[i_eps, i_x]
                    gp = m.gp[i_eps, i_x]
                    jp = itp[χt/gp, xp]
                    expterm += m.Π[i_eps] * (jp * gp)^α
                end
                μ = expterm^(1/α)

                # evaluate value function
                Jupd[i_k, i_x] = ((1 - β)*ct^ρ + β*μ^ρ)^(1/ρ)

            end
        end

        err = maxabs(Jvals - Jupd)
        copy!(Jvals, Jupd)
        itp = scale(interpolate(Jupd, BSpline(Quadratic(Line())), OnGrid()),
                    m.k, m.x)
        @show iter, err, time() - start_time

    end

    return itp
end

end  # module
