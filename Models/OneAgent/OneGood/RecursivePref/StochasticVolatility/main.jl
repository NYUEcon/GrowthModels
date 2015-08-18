module X

using CompEcon

immutable Agent
    ρ::Float64
    α::Float64
    β::Float64
end

_unpack(a::Agent) = (a.ρ, a.α, a.β)

immutable Exog
    A::Float64
    B::Float64
    τ::Float64
    φᵥ::Float64
    vbar::Float64
    p::Matrix{Float64}
    Π::Vector{Float64}
    xp::Matrix{Float64}
    vp::Matrix{Float64}
    gp::Matrix{Float64}
    lgp::Matrix{Float64}
end

function Exog(A::Float64, B::Float64, τ::Float64, φv::Float64, vbar::Float64,
              lgbar::Float64, nϵ1::Int, nϵ2::Int, grid::Matrix{Float64})

    # extract x_t and v_t from grid
    x = grid[:, 2]
    v = grid[:, 3]

    # construct exogenous process
    ϵ, Π = qnwnorm([nϵ1, nϵ2], [0.0, 0.0], eye(2))
    xp = A*x .+ B*sqrt(v).*ϵ[:, 1]'
    vp = (1-φv)*vbar .+ φv*v .+ τ*ϵ[:, 2]'
    gp = exp(lgbar + xp)
    lgp = log(gp)

    # transpose so we can iterate a across states column by column
    xp = xp'
    vp = vp'
    gp = gp'
    lgp = lgp'

    # package
    Exog(A, B, τ, φv, vbar, ϵ, Π, xp, vp, gp, lgp)
end

# ----- #
# Model #
# ----- #
immutable BFZ{BST}
    agent::Agent
    η::Float64
    ν::Float64
    δ::Float64
    grid::Matrix{Float64}
    grid_transpose::Matrix{Float64}  # for iterating state by state over cols
    dim_map::Dict{Symbol,Int}
    basis::Basis{3}
    basis_struc::BasisStructure{BST}
    exog::Exog
end

function BFZ(;ρ::Real=-1, α::Real=-9, β::Real=0.99,
              η::Real=1/3, ν::Float64=0.1, δ::Real=0.025,
              lgbar::Real=0.004, e::Real=1.0, A::Real=0., B::Real=1.,
              vbar::Real=0.015^2, φv::Real=0.95, τ::Real=0.74e-5,
              nk::Int=20, nx::Int=8, nv::Int=8, nϵ1::Int=3, nϵ2::Int=3)

    basis = Basis(SplineParams(collect(linspace(25.0, 50.0, nk)), 0, 3),
                  SplineParams(collect(linspace(-.06, .06, nx)), 0, 1),
                  SplineParams(collect(linspace(1e-4, 4e-4, nv)), 0, 1))

    grid = nodes(basis)[1]

    exog = Exog(A, B, τ, φv, vbar, lgbar, nϵ1, nϵ2, grid)

    basis_struc = BasisStructure(basis, Direct())

    # Create the agent
    agent = Agent(ρ, α, β)

    dim_map = Dict(zip([:k, :x, :v], 1:3))

    # package everything up and return
    BFZ(agent, η, ν, δ, grid, grid', dim_map, basis, basis_struc, exog)
end

_Y(m, k) = (m.η*k.^(m.ν) + (1-m.η)).^(1/m.ν)

function solve_ecm(m::BFZ; ξ::Float64=0.2, tol::Float64=1e-10, maxiter::Int=5000)
    # initialize J
    ρ, α, β = _unpack(m.agent)
    ν, η, δ = m.ν, m.η, m.δ
    M = size(m.grid, 1)
    N = size(m.exog.xp, 1)

    jm_old = (1-β)^(1/ρ)*(_Y(m, m.grid[:, 1]) + (1 - δ)*m.grid[:, 1]) ./ 1000.0
    coefs = funfitxy(m.basis, m.basis_struc, jm_old)[1]

    J_k_basis_struc = BasisStructure(m.basis, Direct(),
                                     nodes(m.basis)[1], [1 0 0])

    err = 1.0
    it = 0

    jp = Array(Float64, N)
    jm = similar(jm_old)
    y_all = _Y(m, m.grid[:, 1])


    while err > tol && it < maxiter
        it += 1
        # compute J at each state
        J = funeval(coefs, m.basis_struc, [0 0 0])
        J_k = funeval(coefs, J_k_basis_struc, [1 0 0])
        for i=1:M
            # extract state and J(m), J_k(m)
            k, x, v = m.grid_transpose[:, i]
            j = J[i]
            jk = J_k[i]
            y = y_all[i]
            gp = m.exog.gp[:, i]

            # solve for c and k'
            c = (jk/(j^(1-ρ) * (1-β) * (y^(1-ν)*η*k + 1 - δ))).^(1/(ρ - 1))
            kp = (y + (1-δ)*k - c) ./ gp

            # Now we need to update the value function. For this we will need
            # to evaluate the spline again at k', x', v' and compute μ
            for n=1:N
                jp[n] = funeval(coefs, m.basis,
                                [kp[n], m.exog.xp[n, i], m.exog.vp[n, i]])[1]
            end
            μ = dot((gp.*jp).^(α), m.exog.Π).^(1/α)
            jm[i] = ((1-β)*c^ρ + β*μ^ρ)^(1/ρ)

        end
        # now let's get ourselves a new coefficient vector
        coef_hat = funfitxy(m.basis, m.basis_struc, jm)[1]
        coefs = (1-ξ)*coef_hat + ξ*coefs
        err = maxabs(jm - jm_old)
        copy!(jm_old, jm)

        @show it, err

    end
end

end  # module
