module TwoAgents

using CompEcon
using NLsolve
include("../ModelAbstraction.jl")
using .ModelAbstraction

immutable BCFL21
    agent1::EpsteinZinAgent
    agent2::EpsteinZinAgent
    production1::CESProducer
    production2::CESProducer
    ac1::AdjCost
    ac2::AdjCost
    exog::ConstantVolatility{2}
    gp::Matrix{Float64}
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
    basis = Basis(SplineParams(collect(linspace(10.0, 40.0, nk1)), 0, 3),
                  SplineParams(collect(linspace(10.0, 40.0, nk2)), 0, 3),
                  SplineParams(collect(linspace(0.1 , 12.0, nU)), 0, 3),
                  LinParams(collect(linspace(0.99, 1.01, nξ)), 0))
    #
    # basis = Basis(ChebParams(nk1, 10.0 , 40.0),
    #               ChebParams(nk2, 10.0 , 40.0),
    #               ChebParams(nU, 1e-2, 12.0),
    #               ChebParams(nξ, 0.99 , 1.01))

    grid, (_k1grid, _k2grid, _Ugrid, _ξgrid) = nodes(basis)

    lξp = (2γ-1)*log(grid[:, 4]) .+ (σ2*exog.ϵ[:, 2]' - σ1*exog.ϵ[:, 1]')
    lξp = clamp(lξp, log(_ξgrid[1]), log(_ξgrid[end]))'
    lgp = (γ-1)*log(grid[:, 4]) .+ σ1*exog.ϵ[:, 1]'

    _en = tuple(map(EndogenousState, (_k1grid, _k2grid, _Ugrid),
                                     (basis[1], basis[2], basis[2]))...)
    _ex = (ExogenousState(_ξgrid, exp(lξp), basis[4]),)

    ss = DefaultStateSpace(_en, _ex, grid, grid', basis)

    # package it up!
    return BCFL21(agent1, agent2, production1, production2, ac1, ac2, exog,
                  exp(lgp)', ss)
end

function get_bs_on_grid(m::BCFL21)
    reduce(CompEcon.row_kron, )
end

_unpack_params(m::BCFL21) = vcat(_unpack(m.agent1)..., _unpack(m.agent2)...,
                             _unpack(m.ac1)..., _unpack(m.ac2)...)

function compute_residuals!(bcfl::BCFL21, state::Vector{Float64}, J::Float64,
                            dJU::Float64, dJk1::Float64, coefs::Vector{Float64},
                            gp::Vector{Float64}, ξp::Vector{Float64},
                            guess::Vector, resid::Vector)

    # Unpack parameters.
    ρ1, α1, β1, ρ2, α2, β2, δ1, η1, δ2, η2 = _unpack_params(bcfl)
    Π = bcfl.exog.Π

    # extract guess
    #I1, I2 = guess[1:2]
    #Up = guess[3:end]
    #AXELLE
    lI1, lI2 = guess[1:2]
    lUp = guess[3:end]
    I1 = exp(lI1)
    I2 = exp(lI2)
    Up = exp(lUp)

    # Get tomorrow's state
    k1, k2, U, ξ = state
    k1p = ((1 - δ1)*k1 + I1 - _ac(bcfl.ac1, k1, I1)) ./ gp
    k2p = ((1 - δ2)*k2 + I2 - _ac(bcfl.ac2, k2, I2)) ./ gp
    statep = [k1p k2p Up ξp]

    # Evaluate value function and its partials at next period's state. This is
    # done at the same time so we only have to compute 7 basis matrices instead
    # of the full 16 we will be using.
    out = funeval(coefs, bcfl.ss.basis, statep,
                  [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0])
    # out will be (4×1×length(gp))
    Jp    = out[:, 1, 1]
    dJpk1 = out[:, 1, 2]
    dJpk2 = out[:, 1, 3]
    dJpU  = out[:, 1, 4]

    # Derivative of Adjustment costs
    dΓ1_dI1 = 1.0 - _dIac(bcfl.ac1, k1, I1)
    dΓ2_dI2 = 1.0 - _dIac(bcfl.ac2, k2, I2)
    dΓ1_dk1 = (1 - δ1) - _dkac(bcfl.ac1, k1, I1)

    # MPK for agent 1. Note we set the `z` arg to one b/c we scaled by z1
    df1dk1 = f_k(bcfl.production1, k1, 1.0)

    # Evaluate all expectations
    μ1 = dot(Π, (gp .* Jp).^(α1))
    μ2 = dot(Π, (gp .* Up).^(α2))
    EV21 = dot(Π, (gp .* Jp).^(α1 - 1.) .* dJpk1)
    EV22 = dot(Π, (gp .* Jp).^(α1 - 1.) .* dJpk2)

    # get consumption
    c1 = (dJk1 - J^(1-ρ1) * μ1^(ρ1-α1) * dΓ1_dk1 * EV21)/(J.^(1-ρ1)*(1-β1)*df1dk1)
    c2 = ((-dJU * U^(1-ρ2) * (1-β2)) / (J^(1-ρ1) * (1-β1) * c1^(ρ1-1)))^(1/(ρ2-1))

    # I residual
    lhsI1 = EV21 .* dΓ1_dI1
    rhsI1 = EV22 .* dΓ2_dI2
    resid[1] = lhsI1 - rhsI1

    # use c1, c2, I1 in budget constraint to get residual for I2
    rhsI2 = c1 + c2 + I1 + I2
    # lhsI2 = bcfl.production1(k1, 1.0) + bcfl.production2(k2, ξ)
    lhsI2 = produce(bcfl.production1, k1, 1.) + produce(bcfl.production2, k2, ξ)
    resid[2] = rhsI2 - lhsI2

    # U residual
    nU = length(U)
    for i=1:nU
        #lhs = -dJU * U.^(1-ρ2) * β2 * μ2.^((ρ2-α2)/α2) .* gp[i]^α2 .* Up[i].^(α2-1)
        #rhs = J^(1-ρ1) * β1 * μ1.^((ρ1-α1)/α1) * gp[i]^α1 * Jp[i]^(α1-1) * (-dJpU[i]) # THIS IS CORRECTED
        # AXELLE
        lhs = log(-dJU * U.^(1-ρ2) * β2 * μ2.^((ρ2-α2)/α2) .* gp[i]^α2 .* Up[i].^(α2-1))
        rhs = log(J^(1-ρ1) * β1 * μ1.^((ρ1-α1)/α1) * gp[i]^α1 * Jp[i]^(α1-1)) + log(-dJpU[i])
        resid[i+1] = lhs - rhs
    end

    return resid
end

function initial_coefs(bcfl::BCFL21, bs::BasisStructure)
    #=
    Here we just manufacturing an initial guess that should be in the ballpark of
    the actual domain and has partial derivatives with the right sign. Specifically,
    we are imposing that:

    - `dJ/dU < 0`
    - `dJ/dk1 > 0`
    - `dJ/dk2 > 0`

    everywhere
    =#

    k1 = bcfl.ss.grid[:, 1]
    k2 = bcfl.ss.grid[:, 2]
    U = bcfl.ss.grid[:, 3]
    ξ = bcfl.ss.grid[:, 4]

    y = produce(bcfl.production1, k1, 1.) + produce(bcfl.production2, k2, ξ)

    # Agent 1's value starts at 1/2 of agent 2's in the "opposite" state (hence the reverse)
    # then gets 1/2 of production added to it.
    J = (reverse(U) + y) / 2.0
    coefs = CompEcon.get_coefs(bcfl.ss.basis, bs, J)

    return coefs
end

function brutal_solution(bcfl::BCFL21; tol=1e-4, maxiter=500)

    # Unpack parameters.
    dist, iter = 10., 0
    ρ1, α1, β1, ρ2, α2, β2, δ1, η1, δ2, η2 = _unpack_params(bcfl)

    # BasisStructure for interpolation
    bs = BasisStructure(bcfl.ss.basis, Direct(), bcfl.ss.grid,
                        [0 0 0 0; 1 0 0 0; 0 0 1 0])

    coefs = initial_coefs(bcfl, bs)

    Nϵ = size(bcfl.gp, 1)

    # give bogus thing here. mean(coefs) ≈ mean(J) (within 1e-2)
    # prev_soln(i::Int) = [0.8*bcfl.ss.grid_transpose[1, i];
    #                     0.8*bcfl.ss.grid_transpose[2, i];
    #                     fill(bcfl.ss.grid_transpose[3, i], Nϵ)]
    # AXELLE
    prev_soln(i::Int) = [log(0.8*bcfl.ss.grid_transpose[1, i]);
                         log(0.8*bcfl.ss.grid_transpose[2, i]);
                         log(fill(bcfl.ss.grid_transpose[3, i], Nϵ))]

    # local out
    while dist > tol && iter < maxiter

        stuff = funeval(coefs, bs, [0 0 0 0; 1 0 0 0; 0 0 1 0])
        J_all = stuff[:, 1, 1]
        dJk1_all = stuff[:, 1, 2]
        dJU_all = stuff[:, 1, 3]

        # function to solve state i
        function ssi(i)
            # prep function for nlsolve
            guess = prev_soln(i)  # TODO
            resid = similar(guess)
            state = bcfl.ss.grid_transpose[:, i]
            J = J_all[i]
            dJk1 = dJk1_all[i]
            dJU = dJU_all[i]
            gp = bcfl.gp[:, i]
            ξp = bcfl.ss.exog[1].gridp[:, i]  # that's a lot of `.`s. True

            # TODO: performance will probably be significantly better if we can outsourse
            #       most of the guts of compute_residuals! elsewhere. Othwerwise we
            #       will be recompiling all the guts for every i on every iteration :yuck:
            function f!(x, fvec)
                 compute_residuals!(bcfl, state, J, dJU, dJk1, coefs, gp, ξp,
                                   x, fvec)
            end

            df = DifferentiableMultivariateFunction(f!)

            # lb = [1e-2; 1e-2; fill(bcfl.ss.grid[1, 3], 4)]
            # ub = [20.0, 20.0, 10.0, 10.0, 10.0, 8.0]
            # mcpsolve(df, guess, lb, ub, iterations=1000, show_trace=true)

            nlsolve(df, guess, iterations=1000, show_trace=true)
        end

        out = map(ssi, 1:size(bcfl.ss.grid, 1))

        # update prev_soln function -- use solution found on this iteration
        prev_soln(i::Int) = out[i].zero

       dist = 0.0  # yay, we are done!
    end

    out

end

end  # module
