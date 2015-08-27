module ModelAbstraction

using CompEcon

# ------------------------------------------------------------------- #
# Economic Entities
# ------------------------------------------------------------------- #
abstract AbstractEntity
abstract AbstractAgent <: AbstractEntity
abstract AbstractConsumer <: AbstractAgent
abstract AbstractProducer <: AbstractAgent
abstract AbstractGovernment <: AbstractAgent  # is the government an entity or an agent?

immutable NotImplementedError <: Exception
    msg::AbstractString
end

# AbstractConsumer Interface
for f in [:is_recursive, :utility]
    @eval $(f){T<:AbstractConsumer}(::T, args...) =
        throw(NotImplementedError("$f not implemented for type $T"))
end

include("MA_Consumers.jl")
include("MA_Producers.jl")

# ------------------------------------------------------------------- #
# State Variables
# ------------------------------------------------------------------- #
abstract AbstractVariable
abstract AbstractStateVariable <: AbstractVariable
abstract AbstractJumpVariable <: AbstractVariable

immutable EndogenousState{T} <: AbstractStateVariable
    grid::Vector{T}
    # FOC::Function
    # env_cond::Function
    # name::Symbol  # could default to `gensym()` (which is julia for I don't care)
end

immutable ExogenousState{T} <: AbstractStateVariable
    grid::Vector{T}  # grid for realizations in the current period
    gridp::Matrix{T}  # grid combining all realizations today with all possible shocks
    # transition::Function
    # name::Symbol
end

# ------------------------------------------------------------------- #
# Exogenous Process
# ------------------------------------------------------------------- #
abstract AbstractExogenousProcess{N}

include("MA_Exogenous.jl")


# ------------------------------------------------------------------- #
# State Spaces
# ------------------------------------------------------------------- #
abstract AbstractStateSpace{N}

export AbstractEntity,
       AbstractConsumer, EpsteinZinAgent, utility, is_recursive,
       AbstractProducer, CESProducer, _production,
       AbstractVariable,
       AbstractStateVariable, EndogenousState, ExogenousState,
       AbstractExogenousProcess,
       ConstantVolatility1, ConstantVolatility, StochasticVolatility, simulate,
       AbstractStateSpace

end
