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

# immutable BFZStateSpace <: AbstractStateSpace{2}
#     k::EndogenousState{Float64}
#     x::ExogenousState{Float64}
#     g::ExogenousState{Float64}
#     grid::Matrix{Float64}
#     grid_transpose::Matrix{Float64}
#     basis::Basis{2}
#     basis_struct::BasisStructure{Direct}
# end

# function BFZStateSpace(lb::Vector=[18.0, -.04, -1e-4],
#                        ub::Vector=[32.0, .04, 1e-4],
#                        n::Vector{Int}=[25, 8, 12],
#                        k::Vector=[3, 1, 1],
#                        bp::Vector=fill(SplineParams, 3),
#                        neps=[3,3])
#     # construct basis
#     basis = Basis(map(bp, lb, ub, n, k)...)

#     # extract grids
#     grid, (kgrid, xgrid, vgrid) = nodes(basis)

#     # TODO: at this point we need to have the exogenous process details also
#     #       in order to construct xp and vp. So maybe that needs to be an
#     #       argument here
#     ks = EndogenousState(kgrid)
# end

export AbstractEntity,
       AbstractConsumer, EpsteinZinAgent, utility, is_recursive,
       AbstractProducer, CESProducer, _production,
       AbstractVariable,
       AbstractStateVariable, EndogenousState, ExogenousState,
       AbstractExogenousProcess,
       ConstantVolatility, StochasticVolatility, simulate,
       AbstractStateSpace

end
