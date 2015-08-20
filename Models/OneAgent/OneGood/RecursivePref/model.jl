module BFZModel

using CompEcon


# ------------------------------------------------------------------- #
# Exogenous Process
# ------------------------------------------------------------------- #
abstract AbstractExogenousProcess

immutable ConstantVolatility <: AbstractExogenousProcess
    A::Float64
    Bv::Float64
    ϵ::Array{Float64, 1}
    Π::Array{Float64, 1}
    Πcumsum::Array{Float64, 1}
end

function ConstantVolatility(A::Float64=0., B::Float64=1., vbar::Float64=0.004, nϵ::Int=5)

    Bv = B * sqrt(vbar)
    ϵ, Π = qnwnorm(nϵ, 0., 1.)
    Πcumsum = cumsum(Π)

    ConstantVolatility(A, Bv, ϵ, Π, Πcumsum)
end

Base.step(ep::ConstantVolatility, x::Float64, ϵ::Float64) = ep.A*x + ep.Bv*ϵ

Base.step(ep::ConstantVolatility, x::Float64) =
    step(ep, x, randn())


immutable StochasticVolatility <: AbstractExogenousProcess
    A::Float64
    B::Float64
    τ::Float64
    φv::Float64
    vbar::Float64
    ϵ::Array{Float64, 2}
    Π::Array{Float64, 1}
    Πcumsum::Array{Float64, 1}
end

function StochasticVolatility(A::Float64=0., B::Float64=1., τ::Float64=7.4e-6,
                              φv::Float64=.95, vbar::Float64=.004,
                              nϵ1::Int=5, nϵ2::Int=5)

    ϵ, Π = qnwnorm([nϵ1, nϵ2], zeros(2), eye(2))
    Πcumsum = cumsum(Π)

    StochasticVolatility(A, B, τ, φ, vbar, ϵ, Π, Πcumsum)
end

function Base.step(ep::StochasticVolatility, xt::Float64, vt::Float64,
                   ϵ1::Float64, ϵ2::Float64)

    xtp1 = ep.A*xt + ep.B*sqrt(vt)*ϵ1
    vtp1 = (1 - ep.φv)*ep.vbar + ep.φv*vt + ep.τ*ϵ2

    return (xtp1, vtp1)
end

Base.step(ep::StochasticVolatility, xt::Float64, vt::Float64) =
    Base.step(ep, xt, vt, randn(2)...)


# ------------------------------------------------------------------- #
# An Epstein-Zin Agent
# ------------------------------------------------------------------- #
abstract Agent

immutable EZAgent
    ρ::Float64
    α::Float64
    β::Float64
end

_unpack(a::EZAgent) = (a.ρ, a.α, a.β)



# ------------------------------------------------------------------- #
# BFZ Model
# ------------------------------------------------------------------- #

end