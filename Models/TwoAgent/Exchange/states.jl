# -------------------------------- #
# Utilities to handle state vector #
# -------------------------------- #
abstract AbstractState{T}

"""
State vector at time t. This is the state vector that can be used to evaluate
the value functions and only contains elements at a single time period.

This is NOT the complete markets state vector used by the planner when updating
the policy function. That vector includes exogenous states at time t+1
"""
type TimeTState{T} <: AbstractState{T}
    l♠::T
    lzbar::T
    v::T
end

"""
Full state vector used by the planner to apply the policy rules.

Includes endogenous and exogenous states at time t as well as exogenous states
at time t+1. We include the t+1 exog states because we have assumed complete
markets, so the planner should be able to choose the allocation based on all
possible shocks.

Notice that we force all arguments to be of the same type. We have constructors
to do conversions as needed
"""
type FullState{T} <: AbstractState{T}
    l♠::T
    lzbar::T
    v::T
    lzbarp::T
    vp::T
    lgp::T
end

@inline function FullState(st::TimeTState, lzbarp::Vector, vp::Vector, lgp::Vector)
    FullState(st.l♠, st.lzbar, st.v, lzbarp, vp, lgp)
end

@inline function FullState(l♠::Real, lzbar::Real, v::Real, lzbarp::Vector,
                           vp::Vector, lgp::Vector)
    N = length(lzbarp)
    FullState(fill(l♠, N), fill(lzbar, N), fill(v, N), lzbarp, vp, lgp)
end

# --------------------------- #
# Interface for AbstractState #
# --------------------------- #

# # Define the iterator protocol for the types
for T in (:TimeTState, :FullState)
    @eval begin
        Base.getindex{T<:Real}(st::$T{Vector{T}}, i::Int) =
            $T([getfield(st, nm)[i] for nm in fieldnames(st)]...)
        Base.length(x::$T) = length(x.l♠)
        Base.size(x::$T) = (length(x.l♠), length(fieldnames(x)))
        Base.start{T<:Real}(x::$T{Vector{T}}) = 1
        Base.next{T<:Real}(x::$T{Vector{T}}, i::Int) = (x[i], i+1)
        Base.done{T<:Real}(x::$T{Vector{T}}, i::Int) = i == length(x)+1
        Base.eltype{T<:Real}(::$T{Vector{T}}) = $T{T}
    end
end

# for easy conversion to Vector and Matrix
Base.convert{T<:Real}(::Type{Vector}, st::AbstractState{T}) =
    vcat([getfield(st, nm) for nm in fieldnames(st)]...)
Base.convert(::Type{Matrix}, st::AbstractState) =
    hcat([getfield(st, nm) for nm in fieldnames(st)]...)

# Convert a FullState to a TimeTState -- just extract first to fields
TimeTState(fst::FullState) = TimeTState(fst.l♠, fst.lzbar, fst.v)
