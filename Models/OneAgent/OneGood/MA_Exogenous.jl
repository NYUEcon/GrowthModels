# ------------------------------------------------------------------- #
# Generic methods
# ------------------------------------------------------------------- #

# Base.size(::AbstractExogenousProcess{N}) = N

# AbstractExogenousProcess Interface
# for f in [:step]
#     @eval $(f){T<:AbstractExogenousProcess}(::T, args...) =
#         throw(NotImplementedError("$f not implemented for type $T"))
# end

# simulate a univariate exog process and return a Vector
function simulate{T<:Number}(ep::AbstractExogenousProcess{1}, x0::T,
                             capT::Int=25_000)
    out = Array(T, capT)
    out[1] = x0
    @inbounds for t=2:capT
        out[t] = step(ep, out[t-1])
    end
    return out
end

function simulate{T<:Number,N}(ep::AbstractExogenousProcess{N},
                               x0::Vector{T}=ones(T, N); capT::Int=25_000)

    out = Array(T, N, capT)
    out[:, 1] = x0
    @inbounds for t=2:capT
        out[:, t] = step(ep, out[:, t-1]...)
    end

    return out
end

# ------------------------------------------------------------------- #
# AR1ConstantVolatility
# ------------------------------------------------------------------- #

immutable ConstantVolatility{N} <: AbstractExogenousProcess{N}
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

    ConstantVolatility{1}(A, Bv, ϵ, Π, Πcumsum)
end

Base.step(ep::ConstantVolatility{1}, x::Float64, ϵ) = ep.A*x .+ ep.Bv*ϵ
Base.step(ep::ConstantVolatility{1}, x::Float64) = step(ep, x, randn())


immutable StochasticVolatility{N} <: AbstractExogenousProcess{N}
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

    StochasticVolatility{2}(A, B, τ, φ, vbar, ϵ, Π, Πcumsum, 2)
end


function Base.step(ep::StochasticVolatility{2}, xt::Float64, vt::Float64, ϵ1, ϵ2)

    xtp1 = ep.A*xt + ep.B*sqrt(vt)*ϵ1
    vtp1 = (1 - ep.φv)*ep.vbar + ep.φv*vt + ep.τ*ϵ2

    return (xtp1, vtp1)
end

Base.step(ep::StochasticVolatility{2}, xt::Float64, vt::Float64) =
    Base.step(ep, xt, vt, randn(2)...)
