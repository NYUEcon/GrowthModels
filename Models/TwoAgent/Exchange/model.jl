module Exchange

using CompEcon

include("util.jl")
using .EDSUtil

# ---------- #
# Agent type #
# ---------- #

immutable Agent
    ρ::Float64
    β::Float64
    α::Float64
    η::Float64
    ν::Float64
    ω::Float64
    σ::Float64
    δ::Float64
    ζ::Float64
end

_unpack(a::Agent) = (a.ρ, a.β, a.α, a.η, a.ν, a.ω, a.σ, a.δ, a.ζ)

# cobb douglass (σ = 0)
# _agg(a::Agent, x, y) = x.^(1-a.ω) .* y.^(a.ω)

# CES
function _agg(agent::Agent, x, y)
    ω, σ = agent.ω, agent.σ
    ((1-ω)*x.^σ + ω*y.^σ).^(1/σ)
end

# ---------- #
# Model type #
# ---------- #

"""
Stores the parameters for the following model:

- 2 Agents with EZ time aggregators:
```tex
U_j = V[c_j, μ_t(U_j')] =[(1-β)c_j^ρ+ β μ_t(U'_j)^ρ]^{1/ρ}
```
- Certainty equivalents:
```tex
μ_t(x) = (E_t[x^α])^{1/α}
```
- Production in coungry j:
```tex
y_j = f(k_j, z_j) = [(1-η)k_j^ν + η z_j^{ν}]^{1/ν}
```
- Aggregators of composite goods
```tex
c_j + i_j = h(a_j, b_j) = [(1-ω)a_j^{σ} + ω b_j^{σ}]^{1/σ}
```
- Resource constarints
```tex
a_j + b_j = y_j = f(k_j, z_j)
```
- Capital law of motion
```tex
k'_j = (1-δ) k_j + i_j = (1-δ) k_j + h(a_j, b_j) - c_j
```
- Shocks
```tex
log z_{1,t+1} = log g + γ log z_{1,t} + (1-γ) log z_{2,t} + ζ_1 w_{1,t+1}
log z_{2,t+1} = log g + γ log z_{2,t} + (1-γ) log z_{1,t} + ζ_2 w_{2,t+1}
w_{1,t+1}, w_{2,t+1} ∼ N(0, 1)
```

Where the parameters (ρ, β, α, η, ν, ω, σ, δ, v) are all potentially different
for each agent (notice that γ is crucially the same across agents)
"""
type BCFL22C
    agent1::Agent
    agent2::Agent
    γ::Float64
    ϵ::Matrix{Float64}
    Π::Vector{Float64}
end

# TODO: check the sigma parameters
function BCFL22C(;ρ1::Real=-1.0 , α1::Real=-9.0, β1::Real=0.999,  # EZ 1
                  η1::Real=1.0, ν1::Real=0.1,                     # production 1
                  σ1::Real=0.6, ω1::Real=0.1,                     # composite 1
                  δ1::Real=0.025, ζ1::Real=0.001,                 # other 1
                  ρ2::Real=-1.0 , α2::Real=-9.0, β2::Real=0.999,  # EZ 2
                  η2::Real=1.0, ν2::Real=0.1,                     # production 2
                  σ2::Real=0.6, ω2::Real=0.1,                     # composite 2
                  δ2::Real=0.025, ζ2::Real=0.001,                 # other 2
                  γ::Real=0.9, nϵ::Int=4)                         # exog
    agent1 = Agent(ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1)
    agent2 = Agent(ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2)
    ϵ, Π = EDSUtil.qnwgh(nϵ, 2)
    BCFL22C(agent1, agent2, γ, ϵ, Π)
end

Base.writemime(io::IO, ::MIME"text/plain", m::BCFL22C) =
    println(io, "BCFL model with 2 goods and 2 agents")

"""
Use this function to generate simulated values for the exogenous state.

Is uses the equation

```tex
log z_{2,t+1} - log z_{1,t+1} = (2γ-1)(log z_{2,t}-log z_{1,t})
                                + w_{2,t+1} - w_{1,t+1}
```

So each increment is normally distributed with mean
(2γ-1)(log z_{2,t} - log z_{1,t}) and variance ζ1 + ζ2
"""
function simulate_exog(m::BCFL22C, capT::Int=10000, seed::Int=1)
    # simplify notation and set random seed
    coef_zbar = (2*m.γ - 1)
    coef_g = (m.γ - 1)
    ζ1, ζ2 = m.agent1.ζ, m.agent2.ζ
    srand(seed)

    # allocate memory and set initial state
    lzbar = zeros(capT)
    lg = zeros(capT)
    lzbar[1] = 0.0

    for t=2:capT
        ϵ1, ϵ2 = randn(), randn()
        old_lzbar = lzbar[t-1]

        lzbar[t] = coef_zbar*old_lzbar + ζ1*ϵ1 - ζ2*ϵ2
        lg[t] = coef_g*old_lzbar + ζ1*ϵ1
    end
    lzbar, lg
end

# ------------------------------ #
# Routines to compute allocation #
# ------------------------------ #
"""
If we combine all 4 FOC and allow all params to be different, we get

```
((1-ω1)*(1-ω2))/(ω1*ω2) = (1-sa)^(σ2-1)/(sa^(σ1-1)) * (sb)^(σ1-1)/((1-sb)^(σ2-1))
```

Given a guess for sa = a1, we can solve the above for b2.

Given a1, b2, zbar=z2/z1 we have

```
a2 = 1 - a1
b1 = zbar - b2
```

The above follows from the fact that in the exchange economy scaled by z1 we
have y1 = 1 and y2 = zbar

Then we can compute c1, c2

Given c1, c2 a1, a2 and stuff we can compute the residual

```
♠ - (c2/a2)^(σ2-1.0) * (c1/a1)^(1.0-σ1) * (1.0-ω1)/ω2
```

or

```
♠ - (c2/b2)^(σ2-1.0) * (c1/b1)^(1.0-σ1) * ω1/(1.0-ω2)
```

The use of either residual is accceptable. Given how we solved for b2, they are
equivalent. However, when a1 is very small, the b2 can be very close to zbar,
making it numerically unstable to divide by b2. So, we prefer the first equation
above
"""
function a1_resid(m::BCFL22C, l♠t, lzbar, a1)
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)

    a2 = 1.0 - a1

    zbar = exp(lzbar)

    if abs(σ1 - σ2) < 1e-14  # sigmas are the same, used closed form sb
        y = ((1-ω1)*(1-ω2)/(ω1*ω2))^(1/(σ1-1))*(a1/a2)
        b2 = zbar / (1 + y)
    else
        error("implement solver for sb")
    end

    # get rest of allocation
    b1 = zbar - b2
    c1 = ((1-ω1)*a1^σ1 + ω1*b1^σ1)^(1/σ1)
    c2 = ((1-ω2)*b2^σ2 + ω2*a2^σ2)^(1/σ2)

    # compute RHS of residual for the a_1 FOC
    rhs = (c2/a2)^(σ2-1.0) * (c1/a1)^(1.0-σ1) * (1.0-ω1)/ω2
    rhs2 = (c2/b2)^(σ2-1.0) * (c1/b1)^(1.0-σ1) * ω1/(1.0-ω2)
    # if abs(rhs - rhs2) > 1e-12
    #     warn("rhs are different for statei l♠t = $(l♠t), zbar=$zbar, a1=$a1")
    #     @show rhs, rhs2
    # end

    # return the residual as first argument and allocation as second
    log(rhs) - l♠t, (a1, a2, b1, b2, c1, c2)
end

function get_allocation(m::BCFL22C, l♠t, lzbart)
    a1t = brent(foo->a1_resid(m, l♠t, lzbart, foo)[1], 1e-15, 1-1e-15)

    # evaluate once more to get the allocation at the optimal a1
    a1t, a2t, b1t, b2t, c1t, c2t = a1_resid(m, l♠t, lzbart, a1t)[2]
end

# ------------------- #
# ValueFunctions type #
# ------------------- #
immutable ValueFunctions
    deg::Int
    coefs::Matrix{Float64}  # first column U_1, second col U_2
end

# return a Vector of [U_1(l♠, zbar), U_2(l♠, lzbar)]. Reshape into vector
value_funcs(vfs::ValueFunctions, l♠::Real, lzbar::Real) =
    reshape(complete_polynomial([l♠ lzbar], vfs.deg) * vfs.coefs, 2)

value_funcs(vfs::ValueFunctions, l♠::Vector, lzbar::Vector) =
    complete_polynomial([l♠ lzbar], vfs.deg) * vfs.coefs

"""
This function assumes you are passing the values of (l♠, zbar) in period t.
It will construct the implied (l♠', zbar', g') for all integration nodes in
m.ϵ and then hand off to the function below
"""
function certainty_equiv(m::BCFL22C, κ::Vector, vfs::ValueFunctions, l♠::Real,
                         lzbar::Real)
    l♠p, lzbarp, lgp = get_next_grid(m, κ, l♠, lzbar)
    certainty_equiv(m, vfs, l♠p, lzbarp, lgp)
end

"""
This method assumes you are passing (l♠', zbar', g') for all the integration
nodes in m.ϵ
"""
function certainty_equiv(m::BCFL22C, vfs::ValueFunctions, l♠p::Vector,
                         lzbarp::Vector, lgp::Vector)
    J = length(m.Π)
    if J != length(l♠p) || J != length(lzbarp) || J != length(lgp)
        error("l♠p, lzbarp and lgp should have $J elements")
    end

    # Compute value functions and evaluate expectations that form certainty
    # equiv
    VF = value_funcs(vfs, l♠p, lzbarp)
    μ1 = dot(m.Π, (exp(lgp).*VF[:, 1]).^(m.agent1.α))^(1.0/m.agent1.α)
    μ2 = dot(m.Π, (exp(lgp).*VF[:, 2]).^(m.agent2.α))^(1.0/m.agent2.α)
    μ1, μ2

end

# TODO: add a "vectorized" version of certainty_equiv that evaluates
#       at mulitple time t states at once. Will help in the euler equation func

# ----------------------- #
# VFI for post simulation #
# ----------------------- #
function _next_grid(m::BCFL22C, κ::Vector, l♠, lzbar, ϵ1, ϵ2)
    # get exog grid
    lzbarp = (2.0*m.γ - 1.0)*lzbar .+ (m.agent1.ζ*ϵ1 - m.agent2.ζ*ϵ2)
    lgp = (m.γ - 1.0)*lzbar .+ m.agent1.ζ*ϵ1

    # use κ to step ♠ forward
    l♠p = κ[1] + κ[2]*l♠ .+ κ[3]*lzbarp + κ[4]*lgp

    return l♠p, lzbarp, lgp
end

"""
Given scalars for ♠ and zbar, return vectors of ♠' and zbar' for every
ϵ' in m.ϵ
"""
function get_next_grid(m::BCFL22C, κ::Vector, l♠::Real, lzbar::Real)
    _next_grid(m, κ, l♠, lzbar, m.ϵ[:, 1], m.ϵ[:, 2])
end

"""
Given vectors for ♠ and zbar, return matrices of ♠' and zbar' for every
ϵ' in m.ϵ.

The layout of the matrix is such that

```
♠[i, j] = ♠'(♠_i, ϵ_j),
```

which means

```
E[♠'] = matrix_from_this_func * m.Π
```
"""
function get_next_grid(m::BCFL22C, κ::Vector, l♠::Vector, lzbar::Vector)
    # NOTE this is the same as the method on scalar l♠ and lzbar except
    # that we traspose the epsilons so broadcasting can make the matrix for us
    _next_grid(m, κ, l♠, lzbar, m.ϵ[:, 1]', m.ϵ[:, 2]')
end

# returns two functions that evaluate U_1(♠, zbar) and U_2(♠, zbar)
function vfi_from_simulation(m::BCFL22C, κ::Vector, grid::Matrix{Float64};
                             deg::Int=3, tol=1e-8, maxit::Int=5000)
    # unpack
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)
    l♠grid, lzbargrid = grid[:, 1], grid[:, 2]
    Ngrid = length(l♠grid)

    # get consumption on the grid
    cs = map((x,y)->get_allocation(m, x, y)[5:6], l♠grid, lzbargrid)
    c1 = map(x->getindex(x, 1), cs)
    c2 = map(x->getindex(x, 2), cs)

    # construct _constant_ basis matrix
    basis_mat = complete_polynomial(grid, deg)

    # get coefficients for degree deg complete poly in ♠ and zbar for the
    # initial guess that U_i = c_i
    coefs = basis_mat \ [c1 c2]
    vfs = ValueFunctions(deg, coefs)

    # advance grid one time step using integration nodes m.ϵ
    l♠p, lzbarp, lgp = get_next_grid(m, κ, l♠grid, lzbargrid)

    # iteration manager stuff + allocate for value functions
    err = 10.0
    it = 0
    U1_old, U2_old = copy(c1), copy(c2)
    U1, U2 = similar(c1), similar(c2)

    # TODO: this slices rows (yuck)
    while err > tol && it < maxit
        it += 1
        for i=1:Ngrid
            # compute certainty equivalents
            μ1, μ2 = certainty_equiv(m, vfs, l♠p[i, :][:], lzbarp[i, :][:],
                                     lgp[i, :][:])

            # apply backward step to get U_{i,t}
            U1[i] = ((1-β1)*c1[i]^ρ1 + β1*μ1^ρ1)^(1/ρ1)
            U2[i] = ((1-β2)*c2[i]^ρ2 + β2*μ2^ρ2)^(1/ρ2)
        end

        # compute residual
        err = max(maxabs(U1 - U1_old), maxabs(U2 - U2_old))

        # update coefficient vector
        coefs = basis_mat \ [U1 U2]
        vfs = ValueFunctions(deg, coefs)

        # update cache of previous U_i
        copy!(U1_old, U1)
        copy!(U2_old, U2)
    end

    vfs
end

# ------------------------- #
# Main simulation algorithm #
# ------------------------- #

#=
now construct LHS of regression using the euler equation:

♠' = ♠ * beta_1/beta_2 * g'^(α1-α2) * (c_1'/c_1)^(ρ_1-1) *
     (c_2'/c_2)^(1-ρ_2) * (U_1'/μ_1)^(α_1 - ρ_1) *(U_2'/μ_2)^(ρ_2 - α_2)
=#
function eval_euler_eqn!(m::BCFL22C, κ::Vector, vfs::ValueFunctions,
                         l♠::Vector, lzbar::Vector, lg_all::Vector, c1::Vector,
                         c2::Vector, capT::Int,
                         LHS::Vector)
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)

    for t=1:capT-1  # can only go to capT-1 b/c we need c[t+1]
        # get μ_t based on current state
        μ1, μ2 = certainty_equiv(m, κ, vfs, l♠[t], lzbar[t])

        # get U_{t+1} from simulation
        U1p, U2p = value_funcs(vfs, l♠[t+1], lzbar[t+1])

        # now package up the LHS of the Euler eqn, which is ♠'
        LHS[t] = (exp(l♠[t])*(β2/β1) *
                  exp(lg_all[t+1])^(α2-α1) *
                  (c1[t+1]/c1[t])^(1.0 - ρ1) *
                  (c2[t+1]/c2[t])^(ρ2 - 1.0) *
                  (U1p/μ1)^(ρ1 - α1) *
                  (U2p/μ2)^(α2 - ρ2))
    end

    # take logs of LHS so it is log♠'
    map!(log, LHS)
end

"""
Do the simulation. Updates c1, c2, ♠, and X in place
"""
function do_simulation!(m::BCFL22C, κ::Vector, capT::Int, lg::Vector,
                        lzbar::Vector, l♠::Vector, c1::Vector, c2::Vector,
                        X::Matrix)
    κ0, κ1, κ2, κ3 = κ

    # simulate ♠ forward and solve for c1, c2 along the way. Need them so
    # I can evaluate the euler equation later
    for t=1:capT
        # extract time t state
        l♠t = l♠[t]
        lzt = lzbar[t]
        lztp = lzbar[t+1]
        lgp = lg[t+1]

        # update ♠_{t+1}. Is linear in ♠_t and log zbar_{t+1}
        l♠p = κ0+ κ1*l♠t + κ2*lztp + κ3*lgp
        l♠[t+1] = l♠p

        # Solve for the optimal allocation at this ♠, zbart and store
        # consumption in vectors
        a1, a2, b1, b2, c1[t], c2[t] = get_allocation(m, l♠t, lzt)

        # fill in this row of regresion matrix
        X[t, 2] = l♠t
    end
end


function linear_coefs(m::BCFL22C, lzbar::Vector{Float64}=simulate_exog(m)[1],
                      lg::Vector{Float64}=simulate_exog(m)[2],
                      κ::Vector{Float64}=[0.0, 0.95, 0.8];
                      maxiter::Int=10_000, grid_skip::Int=5,
                      verbose::Bool=true, print_skip::Int=1,
                      ξ::Float64=0.05)
    # Make sure we are working with the endowment economy
    if abs(m.agent1.η-1.0) > 1e-14 || abs(m.agent2.η - 1.0) > 1e-14
        msg = "Solution only implemented for endowment economy (η1=η2=1.0)"
        error(msg)
    end

    # unpack
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)
    capT = length(lzbar) - 1

    # unpack exog
    γ = m.γ
    ϵ1 = m.ϵ[:, 1]
    ϵ2 = m.ϵ[:, 2]
    Π = m.Π

    # Iteration management stuff
    tol = 1e-4*ξ
    dist = 100.0
    it = 0

    # ♠ is ratio multipliers: φ_2/φ_1. Starts at 1.0, so logs at 0
    l♠ = zeros(capT+1)
    X = [ones(capT) l♠[1:capT] lzbar[2:capT+1] lg[2:capT+1]]  # regression matrix
    LHS = Array(Float64, capT-1)  # LHS of regression from euler equation

    l♠_old = fill(100.0, capT+1)

    c1 = Array(Float64, capT)
    c2 = Array(Float64, capT)

    # core algorithm
    while dist > tol && it < maxiter
        start_time = time()

        # do simulation. Updates ♠, c1, c2, and X inplace
        do_simulation!(m, κ, capT, lg, lzbar, l♠, c1, c2, X)

        # build EDS grid on ♠ and lzbar terms in regression matrix
        if mod(it, grid_skip) == 0 || it == 0
            verbose && info("Updating grid")
            grid = get_eds_grid(X[1:capT-1, 2:3])
        end

        # do VFI to get approximation to value functions/certainty equiv
        vfs = vfi_from_simulation(m, κ, grid; deg=3, tol=1e-8, maxit=5000)

        # update LHS
        eval_euler_eqn!(m, κ, vfs, l♠, lzbar, lg, c1, c2, capT, LHS)

        # compute dist, copy cache, update coefs
        dist = mean(abs(1.0 - exp(l♠ - l♠_old)))
        copy!(l♠_old, l♠)
        κ_hat = X[1:capT-1, :] \ LHS[1:capT-1]
        κ = ξ*κ_hat + (1-ξ)*κ

        it += 1
        if verbose && mod(it, print_skip) == 0
            tot_time = time() - start_time
            @printf "Iteration %i, dist %2.5e, time %5.5e\n" it dist tot_time
            @show [κ κ_hat]'
        end
    end

    l♠, κ

end

function main()
    m = BCFL22C()
    lzbar, lg = simulate_exog(m)
    κ = [0.0, 0.95, -0.95, 0.0]
    ξ = 0.05
    linear_coefs(m, lzbar, lg, κ, maxiter=15)
end

end  # module


#=
function foobar(m::BCFL22C, a1::Real, zbar::Real)
    a2 = 1.0 - a1

    if abs(σ1 - σ2) < 1e-14  # sigmas are the same, used closed form sb
        y = ((1-ω1)*(1-ω2)/(ω1*ω2))^(1/(σ1-1))*(a1/a2)
        b2 = zbar / (1 + y)

        # cobb douglass
        # x = (1-ω1)*(1-ω2)/(ω1*ω2)*(a2/a1)
        # b2 = zbar*x/(1+x)
    else
        error("implement solver for sb")
    end

    # get rest of allocation
    b1 = zbar - b2
    c1 = ((1-ω1)*a1^σ1 + ω1*b1^σ1)^(1/σ1)
    c2 = ((1-ω2)*b2^σ2 + ω2*a2^σ2)^(1/σ2)

    ♠ = (c2/a2)^(1.0-σ2) * (c1/a1)^(σ1-1.0) * ω2/(1-ω1)
    ♠2 = (c2/b2)^(1.0-σ2) * (c1/b1)^(σ1-1.0) * (1-ω2)/ω1

    @show ♠, ♠2
    return ♠

end

function foobar(m::BCFL22C, a1_grid::Vector, zbar_grid::Vector)
    grid = gridmake(a1_grid, zbar_grid)
end
=#
