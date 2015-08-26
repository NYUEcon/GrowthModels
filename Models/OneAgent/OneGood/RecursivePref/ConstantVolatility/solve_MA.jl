module X

using CompEcon
include("../../ModelAbstraction.jl")
using .ModelAbstraction

immutable BFZStateSpace <: AbstractStateSpace{2}
    k::EndogenousState{Float64}
    x::ExogenousState{Float64}
    g::ExogenousState{Float64}
    grid::Matrix{Float64}
    grid_transpose::Matrix{Float64}
    basis::Basis{2}
    basis_struct::BasisStructure{Direct}
end

immutable BFZModel
    agent::EpsteinZinAgent
    producer::CESProducer
    exog::ConstantVolatility
    state_space::BFZStateSpace
end

function BFZcv(;ρ::Real=-1, α::Real=-9, β::Real=0.99,
              η::Real=1/3, ν::Float64=0.1, δ::Real=0.025,
              lgbar::Real=0.004, A::Real=0., B::Real=1., vbar::Real=0.015^2,
              nk::Int=20, nx::Int=8, nϵ1::Int=3)
    # (ρ, α, β, η, ν, δ, lgbar, A, B, vbar, nk, nx, nϵ1) = (-1., -9., .99, .33, .1, .025, .004, 0., 1., .015^2, 20, 8, 9)
    # Create the Consumer
    BFZagent = EpsteinZinAgent(ρ, α, β)

    # Create means of Production
    BFZprod = CESProducer(η, ν, δ)

    # Create Exogenous Process
    exog = ConstantVolatility(A, B, vbar, nϵ1)

    #
    # Create the State Space
    #
    # First Create Basis
    basis = Basis(SplineParams(collect(linspace(15.0, 30.0, nk)), 0, 3),
                  SplineParams(collect(linspace(-.05, .05, nx)), 0, 1))
    basis_struc = BasisStructure(basis, Direct())

    # Get the implied grids
    grid, (kgrid, xgrid) = nodes(basis)
    grid_transpose = grid'
    ggrid = exp(lgbar + xgrid)

    xpgrid = Array(Float64, nϵ1, nx)
    gpgrid = Array(Float64, nϵ1, nx)
    for i=1:nx
        xpgrid[:, i] = step(exog, xgrid[i], exog.ϵ)
        gpgrid[:, i] = exp(lgbar + xpgrid[:, i])
    end

    # Create the states
    kstate = EndogenousState(kgrid)
    xstate = ExogenousState(xgrid, xpgrid)
    gstate = ExogenousState(ggrid, gpgrid)

    # Finally create space
    statespace = BFZStateSpace(kstate, xstate, gstate, grid, grid_transpose,
                               basis, basis_struc)

    return BFZModel(BFZagent, BFZprod, exog, statespace)
end


function envelope_method(bfz::BFZModel; maxiter=500, tol=1e-4)

    # Useful parameters
    dist = 10.
    iter = 1
    (ρ, α, β) = bfz.agent.ρ, bfz.agent.α, bfz.agent.β
    (η, ν, δ) = bfz.producer.η, bfz.producer.ν, bfz.producer.δ
    ρinv = 1./ρ

    # Useful function
    _Y(k) = _production(bfz.producer, k)

    # Pull information out of type
    kgrid = bfz.state_space.k.grid
    xgrid = bfz.state_space.x.grid
    stategrid = bfz.state_space.grid
    nk, nx = length(kgrid), length(xgrid)

    # Initialize Value Function and Interpolant
    JT = (((1 - β) * (_Y(stategrid[:, 1])+(1 - δ)*stategrid[:, 1]).^(ρ)).^(ρinv)) ./ 500
    Jvals = repeat(JT, inner=[1, nx])
    Jupd = copy(Jvals)
    Jcoeffs = funfitxy(bfz.state_space.basis, bfz.state_space.basis_struct)
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
                Jt = evaluate(Jspl, kt, xt)
                ct = (dJ_k/((1-β)*Jt^(1-ρ)*(yt^(1-ν)*η*kt^(ν-1)+1-δ)))^(1./(ρ-1))
                χt = yt + (1 - δ)*kt - ct
                if χt < 0
                    warn("Negative chi")
                end
                χpolicy[kind, xind] = χt
                expterm = 0.

                for shockind=1:nshocks
                    xtp1 = xpgrid[shockind, xind]
                    gtp1 = gpgrid[shockind, xind]
                    jtp1 = evaluate(Jspl, χt/gtp1, xtp1)
                    expterm += (jtp1 * gtp1)^α * Π[shockind]
                end
                μ = expterm^αinv

                Jupd[kind, xind] = ((1 - β)*ct^ρ + β*μ^ρ)^(ρinv)

            end
        end

        dist = maxabs(Jvals - Jupd)
        copy!(Jvals, Jupd)
        Jspl = Spline2D(kgrid, xgrid, Jvals; kx=1, ky=1, s=0.0)
        @show iter, dist

    end

    return Jvals, χpolicy
end


end  # End Module