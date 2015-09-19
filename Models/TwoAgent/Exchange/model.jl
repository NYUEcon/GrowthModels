module Exchange

using CompEcon
using Distances

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

function _agg(agent::Agent, a, b)
    ω, σ = agent.ω, agent.σ
    ((1-ω)*a.^σ + ω*b.^σ).^(1/σ)
end

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
end

# TODO: check the sigma parameters
function BCFL22C(;ρ1::Real=-1.0 , α1::Real=-9.0, β1::Real=0.999,  # EZ 1
                  η1::Real=1.0, ν1::Real=0.1,                     # production 1
                  σ1::Real=0.6, ω1::Real=0.7,                     # composite 1
                  δ1::Real=0.025, ζ1::Real=0.001,                 # other 1
                  ρ2::Real=-1.0 , α2::Real=-9.0, β2::Real=0.999,  # EZ 2
                  η2::Real=1.0, ν2::Real=0.1,                     # production 2
                  σ2::Real=0.6, ω2::Real=0.7,                     # composite 2
                  δ2::Real=0.025, ζ2::Real=0.001,                 # other 2
                  γ::Real=0.9)                                    # exog
    agent1 = Agent(ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1)
    agent2 = Agent(ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2)
    BCFL22C(agent1, agent2, γ)
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
(2γ-1)(log z_{2,t} - log z_{1,t}) and variance v1 + v2

Arguments:

- `m::BCFL21`: The model containing the exogenous proces to simulate
- `keepers::StepRange{Int,Int}`: A range of the form `start:thin:end`
  that specifies that the simulation will run for `end - start` periods and
  every `thin` observations will be kept.
"""
function simulate_exog(m::BCFL22C, keepers::StepRange{Int,Int}=1:1:10000,
                       seed::Int=1)
    # simplify notation and set random seed
    coef = (2*m.γ - 1)
    ζ = m.agent1.ζ + m.agent2.ζ
    srand(seed)

    # allocate memory and set initial state
    N_out = length(keepers)
    out = zeros(N_out)
    out[1] = 0.0
    step_length = step(keepers)
    temp = 0.0

    for keep=2:N_out
        for _ in step_length
            temp = coef*temp + ζ*randn()
        end
        out[keep] = temp
    end
    out
end

function get_a_b_c(m::BCFL22C, sa, sb, zbar)
    a1 = sa
    a2 = 1-sa
    b1 = sb*zbar
    b2 = (1-sb)*zbar

    c1 = _agg(m.agent1, a1, b1)
    c2 = _agg(m.agent2, a2, b2)

    return a1, a2, b1, b2, c1, c2
end

"""
If we combine all 4 FOC and allow all params to be different, we get

```
((1-ω1)*(1-ω2))/(ω1*ω2) = (1-sa)^(σ2-1)/(sa^(σ1-1)) * (sb)^(σ1-1)/((1-sb)^(σ2-1))
```

Given a guess for sa, we can solve the above for sb.

Given sa, sb, zbar=z2/z1 we have

```
a1 = sa
a2 = 1-sa
b1 = sb*zbar
b2 = (1-sb)*zbar
```

The above follows from the fact that in the exchange economy scaled by z1 we
have y1 = 1 and y2 = zbar

Then we can compute c1, c2

Given c1, c2 a1, a2 and stuff we can compute the residual

```
♠ - (1-β2)*c2^(ρ2-σ2)*(1-ω2)*b2^(σ2-1)/((1-β1)*c1^(ρ1-σ1)*ω1*b1^(σ1-1))
```

or

```
♠ - (1-β2)*c2^(ρ2-σ2)*ω2*a2^(σ2-1)/((1-β1)*c1^(ρ1-σ1)*(1-ω1)*a1^(σ1-1))
```

It isn't obvious which of the two residuals to use. In principle they should be
the same at the solution.

"""
function sa_resid(m::BCFL22C, ♠t, zbar, sa)
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)

    if abs(σ1 - σ2) < 1e-14  # sigmas are the same, used closed form sb
        x = ((1-ω1)*(1-ω2)/(ω1*ω2))^(1/(σ1-1))*(sa/(1-sa))
        sb = x / (1 + x)
    else
        error("implement solver for sb")
    end

    a1, a2, b1, b2, c1, c2 = get_a_b_c(m, sa, sb, zbar)

    rhs = (1-β2)*c2^(ρ2-σ2)*ω2*a2^(σ2-1)/((1-β1)*c1^(ρ1-σ1)*(1-ω1)*a1^(σ1-1))
    return ♠t - rhs, (sb, a1, a2, b1, b2, c1, c2)
end


function linear_coefs(m::BCFL22C, lzbar::Vector{Float64}=simulate_exog(m),
                      b_old::Vector{Float64}=[0.05, 0.95, -0.75];
                      maxiter::Int=10_000, verbose::Bool=true,
                      ξ::Float64=0.1)
    # Make sure we are working with the endowment economy
    if abs(m.agent1.η-1.0) > 1e-14 || abs(m.agent2.η - 1.0) > 1e-13
        m = "Solution only implemented for endowment economy (η1=η2=1.0)"
        error(m)
    end

    # unpack
    ρ1, β1, α1, η1, ν1, ω1, σ1, δ1, ζ1 = _unpack(m.agent1)
    ρ2, β2, α2, η2, ν2, ω2, σ2, δ2, ζ2 = _unpack(m.agent2)
    capT = length(lzbar)

    # Iteration management stuff
    tol = 1e-4*ξ
    dist = 100.0
    it = 0

    # ratio of pareto weights: agent 1/agent 2. Starts at 1.0. Is length T+1 so
    # we can run the for loop below to fill in the entire X/RHS arrays. We will
    # not use the last point of ♠
    ♠ = ones(capT+1)
    X = [ones(capT) ♠[1:capT] lzbar]  # regression matrix
    RHS = Array(Float64, capT)  # residual to use in regression

    ♠_old = fill(100.0, capT+1)

    while dist > tol

        showplot(plot(1:capT, ♠[1:capT]))

        b0, b1, b2 = b_old
        t = 1
        # simulate ♠ forward and solve for sa along the way
        @inbounds for t=1:capT
            # extract time t state
            ♠t = ♠[t]
            lzt = lzbar[t]
            zt = exp(lzt)

            # update ♠_{t+1}
            ♠p = b0+ b1*♠t + b2*lzt
            ♠[t+1] = ♠p

            sa_resid(m, ♠t, zt, 1-1e-13)[1]

            # compute sa_t as a function of ♠_t and zbar_t
            sa = brent(foo->sa_resid(m, ♠t, zt, foo)[1], 1e-15, 1-1e-15)

            # evaluate once more to get all the other intermediate variables
            # at the sa optimum
            sb, a1, a2, b1, b2, c1, c2 = sa_resid(m, ♠t, zt, sa)[2]

            # fill in RHS (other residual we didn't use in sa_resid above)
            # we don't need to do any transformations because the LHS is
            # already supposed to represent ♠t
            RHS_num = (1-β2)*c2^(ρ2-σ2)*(1-ω2)*b2^(σ2-1)
            RHS_denom = (1-β1)*c1^(ρ1-σ1)*ω1*b1^(σ1-1)
            RHS[t] = RHS_num / RHS_denom

            # also fill in endog part of regression matrix for time t
            X[t, 2] = ♠t
        end

        # check convergence (computes `mean(abs(♠[1:end-1] - ♠_old[1:end-1]))`)
        # dist = cityblock(sub(♠, 1:capT), sub(♠_old, 1:capT)) / capT
        #
        # # Now update RHS inplace to be b_hat, then update coefs
        # b_hat = A_ldiv_B!(qrfact(X, Val{true}), RHS)[1:3]
        # b_old = scale!(ξ, b_hat) + scale!((1-ξ), b_old)

        dist = mean(abs(1 - ♠./♠_old))
        b_hat = X \ RHS
        b_old = ξ*b_hat + (1-ξ)*b_old

        copy!(♠_old, ♠)

        it += 1
        if mod(it, 1) == 0
            @printf "Iteration %i, dist %2.5e\n" it dist
            @show b_old
        end
    end

    ♠, b_old

end

#=
What I need to do to implement the GSSA algorithm:

- Simulate exog state. For me this is xi
- Initial guess for the endogenous state
- Guess initial policy rules
- Repeat the following:
    - Use policy rules to simulate
=#
end  # module
