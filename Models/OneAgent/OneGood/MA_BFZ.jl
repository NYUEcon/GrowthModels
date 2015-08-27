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


end
