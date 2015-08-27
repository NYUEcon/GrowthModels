module TwoAgents

using CompEcon
include("../ModelAbstraction.jl")
using .ModelAbstraction

immutable BFCL21
    agent1::EpsteinZinAgent
    agent2::EpsteinZinAgent
    production1::CESProducer
    production2::CESProducer
    ac1::AdjCost
    ac2::AdjCost
    exog::ConstantVolatility{2}
    lgp::Matrix{Float64}
    ss::DefaultStateSpace{3,1,Float64,4}
end

function BCFL21(;ρ1::Real=-1.0 , α1::Real=-9.0, β1::Real=0.999,
                 ρ2::Real=-1.0 , α2::Real=-9.0, β2::Real=0.999,
                 η1::Real=1.0/3, ν1::Real=.1, η2::Real=1.0/3, ν2::Real=.1,
                 γ::Real=.9, δ::Real=0.025, σ1::Real=0.001, σ2::Real=0.001,
                 nϵ1::Int=2, nϵ2::Int=2,
                 nk1::Int=10, nk2::Int=10, nU::Int=8, nξ::Int=5)

    # Agents
    agent1 = EpsteinZinAgent(ρ1, α1, β1)
    agent2 = EpsteinZinAgent(ρ2, α2, β2)

    # Producers
    production1 = CESProducer(η1, ν1)
    production2 = CESProducer(η2, ν2)

    # AdjustmentCosts
    ac1 = AdjCost(δ, .75)
    ac2 = AdjCost(δ, .75)

    # exog
    exog = ConstantVolatility(γ, σ1, σ2, nϵ1, nϵ2)

    # state space
    basis = Basis(SplineParams(collect(linspace(10.0, 40.0, nk1), 0, 3)),
                  SplineParams(collect(linspace(10.0, 40.0, nk2), 0, 3)),
                  SplineParams(collect(linspace(1e-2 , 25.0, nU), 0, 3)),
                  LinParams(collect(linspace(0.99, 1.01, nξ), 0)))

    grid, (_k1grid, _k2grid, _Ugrid, _ξgrid) = nodes(basis)

    lξp = (2γ-1)*log(grid[:, 4]) .+ (σ2*exog.ϵ[:, 2]' - σ1*exog.ϵ[:, 1]')
    lξp = clamp(lξp, log(_ξgrid[1]), log(_ξgrid[end]))
    lgp = (γ-1)*log(grid[:, 4]) .+ σ1*ϵ[:, 1]'  # where will this live?

    endog = tuple(map(EndogenousState, (_k1grid, _k2grid, _Ugrid),
                                       (basis[1], basis[2], basis[2]))...)
    exog = (ExogenousState(_ξgrid, exp(lξp), basis[4]))
    ss = DefaultStateSpace(endog, exog, grid, basis)

    # package it up!
    return BFCL21(agent1, agent2, production1, production2, ac1, ac2, exog,
                  lgp, ss)
end

_unpack_params(m::BCFL21) = (_unpack(m.agent1)..., _unpack(m.agent2)...,
                             _unpack(m.ac1)..., _unpack(m.ac2))

function compute_residuals(bfcl::BFCL21, state::Vector{Float64},
                           J_coefs::Vector{Float64}, J::Float64, dJU::Float64,
                           guess::Vector{Float64})

    # Unpack parameters.
    ρ1, α1, β1, ρ2, α2, β2, δ1, ikbar1, δ2, ikbar2 = _unpack_params(bcfl)

    # extract guess
    I1, I2 = guess[1:2]
    Up = guess[3:end]

    # Get tomorrow's state
    # TODO: don't we need to worry about gp here also?
    k1, k2, U, ξ = state
    k1p = (1 - δ1)*k1t + I1 - _ac(bfcl.ac1, k1t, I1)
    k2p = (1 - δ2)*k2t + I2 - _ac(bfcl.ac2, k2t, I2)
    statep = [k1p k2p Up ξp]

    # Evaluate value function and its partials at next period's state. This is
    # done at the same time so we only have to compute 7 basis matrices instead
    # of the full 16 we will be using.
    out = funeval(Jcoeffs, bcfl.basis, statep,
                  [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0])
    Jp    = out[:, :, 1]
    dJpk1 = out[:, :, 2]
    dJpk2 = out[:, :, 3]
    dJpU  = out[:, :, 4]

    # Derivative of Adjustment costs
    dGdI1 = _dIac(bcfl.ac1, k1t, I1)
    dGdI2 = _dIac(bcfl.ac2, k2t, I2)
    dGdk1 = _dkac(bcfl.ac1, k1t, I1)

    # MPK
    df1dk1 = f_k(bcfl.production1, k1)

    # Evaluate all expectations
    μ1 = dot(Π, (gp .* Jp).^(α1))
    μ2 = dot(Π, (gp .* Up).^(α2))
    EV21 = dot(Π, (gp .* Jp).^(α1 - 1.) .* dJpk1)
    EV22 = dot(Π, (gp .* Jp).^(α1 - 1.) .* dJpk2)

    # I residual
    lhsI = EV21 .* dGdI1
    rhsI = EV22 .* dGdI2
    residI = lhsI - rhsI

    c1 = (dJpk1 - J^(1-ρ1) μ1^(ρ1-α1) * dGdk1 * EV21)/(J.^(1-ρ1)*(1-β1)*df1dk1)
    c2 = (-)

    # U residual
    nU = length(U)
    residUp = Array(Float64, nU)
    for i=1:nU
        lhs = -dJU * U.^(1 - ρ2) * β2 * μ2.^((ρ2 - α2)/α2) .* gp[i]^α2 .* Up[i].^(α2-1)
        rhs = J^(1 - ρ1) * β1 * μ1.^((ρ1 - α1)/α1) * gp[i]^α1 * Jp[i]^(α1-1) * dJpU[i]
        residUp[i] = lhs - rhs
    end


    return [residI; residUp]
end

function brutal_solution(bfcl::BFCL21; tol=1e-4, maxiter=500)

    nothing
end

end  # module
