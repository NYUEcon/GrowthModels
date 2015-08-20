module X

using CompEcon

immutable EZAgent
    ρ::Float64
    α::Float64
    β::Float64
end

_unpack(a::EZAgent) = (a.ρ, a.α, a.β)

immutable Exog
    A::Float64
    B::Float64
    τ::Float64
    φᵥ::Float64
    vbar::Float64
    lgbar::Float64
    ϵp::Matrix{Float64}
    Π::Vector{Float64}
    xp::Matrix{Float64}
    vp::Matrix{Float64}
    gp::Matrix{Float64}
end

function step(A::Float64, B::Float64, φᵥ::Float64, vbar::Float64, τ::Float64, x,
              v, ϵ1, ϵ2)
    xp = A*x .+ B*sqrt(v).*ϵ1
    vp = (1-φᵥ)*vbar .+ φᵥ*v .+ τ.*ϵ2
    xp, vp
end

step(ex::Exog, x, v, ϵ1=ex.ϵp[:, 1], ϵ2=ex.ϵp[:, 2]) =
    step(ex.A, ex.B, ex.φᵥ, ex.vbar, ex.τ, x, v, ϵ1, ϵ2)

function Exog(A::Float64, B::Float64, τ::Float64, φᵥ::Float64, vbar::Float64,
              lgbar::Float64, nϵ1::Int, nϵ2::Int, grid::Matrix{Float64})

    # extract x_t and v_t from grid
    x = grid[:, 2]
    v = grid[:, 3]

    # construct exogenous process
    ϵp, Π = qnwnorm([nϵ1, nϵ2], [0.0, 0.0], eye(2))
    xp, vp = step(A, B, φᵥ, vbar, τ, x, v, ϵp[:, 1]', ϵp[:, 2]')
    gp = exp(lgbar + xp)

    # package -- note, transpose (⋅)p variables so 1 state = 1 column
    Exog(A, B, τ, φᵥ, vbar, lgbar, ϵp, Π, xp', vp', gp')
end

function simulate(ex::Exog; T::Int=25_000, v0::Float64=0.001, x0::Float64=0.0)
    A, B, φᵥ, τ, vbar = ex.A, ex.B, ex.φᵥ, ex.τ, ex.vbar
    v = Array(Float64, T)
    x = Array(Float64, T)

    # set initial conditions
    v[1] = v0
    x[1] = x0

    # constant in law of motion for v
    c = (1 - φᵥ) * vbar

    @inbounds for t=1:T-1
        x[t+1], v[t+1] = step(ex, x[t], v[t], rand(), rand())
    end

    x, v
end


# ----- #
# Model #
# ----- #

immutable BFZ
    agent::EZAgent
    η::Float64
    ν::Float64
    δ::Float64
    grid::Matrix{Float64}
    grid_transpose::Matrix{Float64}  # for iterating state by state over cols
    basis::Basis{3}
    basis_struc::BasisStructure
    exog::Exog
end

function BFZ(;ρ::Real=-1, α::Real=-9, β::Real=0.9995,
              η::Real=1/3, ν::Float64=0.1, δ::Real=0.025,
              lgbar::Real=0.004, e::Real=1.0, A::Real=0., B::Real=1.,
              vbar::Real=0.01^2, φv::Real=0.975, τ::Real=(vbar/3)*sqrt(1-φv^2),
              nk::Int=20, nx::Int=8, nv::Int=8, nϵ1::Int=3, nϵ2::Int=3)

    basis = Basis(SplineParams(collect(linspace(18.0, 32.0, nk)), 0, 3),
                  SplineParams(collect(linspace(-.04, .04, nx)), 0, 1),
                  SplineParams(collect(linspace(1e-4, 4e-4, nv)), 0, 1))

    grid = nodes(basis)[1]

    exog = Exog(A, B, τ, φv, vbar, lgbar, nϵ1, nϵ2, grid)

    basis_struc = BasisStructure(basis, Direct())

    # Create the agent
    agent = EZAgent(ρ, α, β)

    # special case for ν. Make sure it isn't negative and then clamp it to be no
    # smaller than 1e-6. ν=0 ⟶ production is cobb-Douglass. Numerically we can't
    # let ν=0, but if we set it very close to zero then we approximate
    # Cobb-Douglass really well. With some testing I found 1e-8 was optimal
    ν < 0 && throw(ArgumentError("ν cannot be negative"))
    ν = clamp(ν, 1e-8, Inf)

    # package everything up and return
    BFZ(agent, η, ν, δ, grid, grid', basis, basis_struc, exog)
end

_Y(m, k) = (m.η*k.^(m.ν) + (1-m.η)).^(1/m.ν)
_VF(m, c, μ) = ((1-m.agent.β)*c^m.agent.ρ + β*μ^m.agent.ρ)^(1/m.agent.ρ)
_CE(m, jp, gp) = dot((gp.*jp).^(m.agent.α), m.exog.Π).^(1/m.agent.α)

function solve_this_state(m::BFZ, i::Int, xv_mat::AbstractMatrix, coefs::Vector,
                          y::Real, j::Real, jk::Real, gp::Vector)
    k, x, v = m.grid_transpose[:, i]
    ρ, α, β = _unpack(m.agent)
    ν, η, δ = m.ν, m.η, m.δ

    N = length(gp)

    # solve for c and k'
    c = (jk/(j^(1-ρ) * (1-β) * (y^(1-ν)*η*k^(ν-1) + 1 - δ))).^(1/(ρ - 1))
    kp = (y + (1-δ)*k - c) ./ gp

    # compute j' to evaluate μ
    kp_basis_mat = CompEcon.evalbase(m.basis.params[1], kp, 0)[1]
    jp = CompEcon.row_kron(xv_mat, kp_basis_mat) * coefs

    μ = _CE(m, jp, gp)
    jm = _VF(m, c, μ)

    return jm, c, kp
end

function basis_mat_xv(m::BFZ, x::Vector, v::Vector)
    mat_v = CompEcon.evalbase(m.basis.params[3], v, 0)[1]
    mat_x = CompEcon.evalbase(m.basis.params[2], x, 0)[1]
    CompEcon.row_kron(mat_v, mat_x)::typeof(mat_v)
end

function solve_ecm(m::BFZ; ξ::Float64=0.4, tol::Float64=1e-10, maxiter::Int=5000)
    # initialize J
    ρ, α, β = _unpack(m.agent)
    ν, η, δ = m.ν, m.η, m.δ
    M = size(m.grid, 1)
    N = size(m.exog.xp, 1)

    jm_old = (1-β)^(1/ρ)*(_Y(m, m.grid[:, 1]) + (1 - δ)*m.grid[:, 1]) ./ 1000.0
    coefs = funfitxy(m.basis, m.basis_struc, jm_old)[1]

    J_k_basis_struc = BasisStructure(m.basis, Direct(),
                                     nodes(m.basis)[1], [1 0 0])

    xv_mats = [basis_mat_xv(m, m.exog.xp[:, i], m.exog.vp[:, i]) for i=1:M]
    err = 1.0
    it = 0

    jm = similar(jm_old)
    y_all = _Y(m, m.grid[:, 1])

    local out  # declare out local so we can return it after the loop
    while err > tol && it < maxiter
        it += 1
        # compute J at each state
        J = funeval(coefs, m.basis_struc, [0 0 0])
        J_k = funeval(coefs, J_k_basis_struc, [1 0 0])

        # solve at each state in parallel
        ssi(i) = solve_this_state(m, i, xv_mats[i], coefs, y_all[i], J[i],
                                  J_k[i], m.exog.gp[:, i])
        out = map(ssi, 1:M)

        # make sure we have jm::Vector{Float64}
        jm = Float64[i[1] for i in out]

        # now let's get ourselves a new coefficient vector
        coef_hat = funfitxy(m.basis, m.basis_struc, jm)[1]
        coefs = (1-ξ)*coef_hat + ξ*coefs
        err = maxabs(jm - jm_old)
        copy!(jm_old, jm)

        @show it, err

    end
    return coefs, out
end

function main()
    m = BFZ()
    solve_ecm(m)
end

function simulate_solution(m::BFZ, coefs::Vector, c::Vector; T::Int=75_000,
                           burn::Int=floor(Int, T/10))
    # simulate exog process
    x, v = simulate(m.exog; T=T)

    # allocate memory for simulated c and k
    c_sim = Array(Float64, T-1)
    k_sim = Array(Float64, T)
    y_sim = Array(Float64, T-1)

    # "arbitrary" initial condition for k. Mean of k on the grid
    k_sim[1] = mean(m.grid[:, 1])

    # get coefficients for approximation of c
    c_coefs = get_coefs(m.basis, m.basis_struc, c)

    gbar = exp(m.exog.lgbar)

    # simulate c, j forward
    @inbounds for t=1:T-1
        c_sim[t] = funeval(c_coefs, m.basis, [k_sim[t] x[t] v[t]])[1]
        gp = gbar*exp(x[t+1])
        y_sim[t] = _Y(m, k_sim[t])
        k_sim[t+1] = (y_sim[t] + (1-m.δ)*k_sim[t] - c_sim[t]) / gp
    end
    x, v, c_sim, k_sim, y_sim = (x[burn:end], v[burn:end], c_sim[burn:end],
                                 k_sim[burn:end], y_sim[burn:end])

    c_coefs, x, v, c_sim, k_sim, y_sim
end

function euler_errors_sim(m::BFZ, coefs::Vector, c_coefs::Vector, x_sim, v_sim,
                          c_sim, k_sim, y_sim)
    ρ, α, β = _unpack(m.agent)
    ν, δ, η = m.ν, m.δ, m.η

    # array to hold the errors, which are computed by solving
    # (c*(1-err))^{ρ-1} = RHS for err ⟶ err = 1- RHS^{1/(ρ-1)}/c
    err_sim = similar(c_sim)

    gbar = exp(m.exog.lgbar)

    for t=1:length(c_sim)-1
        # extract stuff for this state
        ct, kt, yt, xt, vt = c_sim[t], k_sim[t], y_sim[t], x_sim[t], v_sim[t]

        # push exog forward using quadrature nodes
        xp, vp = step(m.exog, xt, vt)
        gp = gbar*exp(xp)

        # get kp; then yp, cp, and jp; then μ
        kp = (yt + (1-m.δ)*kt - ct) ./ gp
        yp = _Y(m, kp)
        bs = BasisStructure(m.basis, Direct(), [kp xp vp], [0 0 0])
        cp = funeval(c_coefs, bs, [0 0 0])
        jp = funeval(coefs, bs, [0 0 0])
        μ = _CE(m, jp, gp)

        # now I can eval RHS
        to_E = gp.^(α-1) .* jp.^(α-ρ) .* cp.^(ρ-1) .* (η*yp.^(1-ν).*kp.^(ν-1)+1-δ)
        RHS = β * μ^(ρ-α) * dot(to_E, m.exog.Π)

        # now fill in err_sim
        err_sim[t] = 1 - RHS^(1/(ρ-1))/ct

    end
    err_sim
end

function euler_errors_ll(m::BFZ, x_sim, v_sim)



end  # module
