# ------------------------------------------------------------------- #
# Epstein Zin
# ------------------------------------------------------------------- #
immutable EpsteinZinAgent <: AbstractConsumer
    ρ::Float64
    α::Float64
    β::Float64
end
_unpack(ez::EpsteinZinAgent) = (ez.ρ, ez.α, ez.β)

is_recursive(a::EpsteinZinAgent) = a.ρ == a.α ? false : true

utility(a::EpsteinZinAgent, flow, continuation) =
    ((1 - a.β).*flow.^a.ρ + a.β * continuation.^a.ρ).^(1./a.ρ)

# ------------------------------------------------------------------- #
# Power Utility
# ------------------------------------------------------------------- #
immutable CRRAAgent <: AbstractConsumer
    γ::Float64
    β::Float64
end

typealias PowerUtilityAgent CRRAAgent

is_recursive(::CRRAAgent) = false
utility(a::CRRAAgent, c) = c.^(1-a.γ)./(1 - a.γ)
