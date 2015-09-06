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
    msg::String
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

immutable EndogenousState{T,BF,BP,TM} <: AbstractStateVariable
    grid::Vector{T}
    basis::Basis{1,BF,BP}
    Φ::TM
    # name::Symbol  # could default to `gensym()` (which is julia for I don't care)
end

EndogenousState(grid::Vector) =
    EndogenousState(grid, Basis(LinParams(grid, 0)))

# Note depending on the type of Basis you pass, the grid inside the state
# instance might be different than the grid you passed in.
function EndogenousState(grid::Vector, b::Basis{1})
    real_grid = nodes(b)[1]
    Φ = CompEcon.evalbase(b.params[1], real_grid, 0)[1]
    EndogenousState(real_grid, b, Φ)
end

immutable ExogenousState{T,BF,BP,TM} <: AbstractStateVariable
    grid::Vector{T}   # grid for realizations in the current period
    gridp::Matrix{T}  # grid moving each i ∈ grid  with all possible shocks
    basis::Basis{1,BF,BP}
    Φ::TM
    # name::Symbol
end

ExogenousState{T}(grid::Vector{T}, gridp::Matrix{T}) =
    ExogenousState(grid, gridp, Basis(LinParams(grid, 0)))

function ExogenousState(grid::Vector, gridp, b::Basis{1})
    real_grid = nodes(b)[1]
    if real_grid != grid
        msg = string("You must use `grid = nodes(b)` and compute `gridp`",
                     " based on that grid or things won't line up")
        throw(ArgumentError(msg))
    end
    Φ = CompEcon.evalbase(b.params[1], real_grid, 0)[1]
    ExogenousState(real_grid, gridp, b, Φ)
end

# ------------------------------------------------------------------- #
# Exogenous Process
# ------------------------------------------------------------------- #
abstract AbstractExogenousProcess{N}

include("MA_Exogenous.jl")

# ------------------------------------------------------------------- #
# State Spaces
# ------------------------------------------------------------------- #
abstract AbstractStateSpace{Nendog,Nexog}

immutable DefaultStateSpace{Nendog,Nexog,T,N} <: AbstractStateSpace{N}
    endo::NTuple{Nendog,EndogenousState{T}}
    exog::NTuple{Nexog,ExogenousState{T}}
    grid::Matrix{T}
    grid_transpose::Matrix{T}
    basis::Basis{N}
end

function DefaultStateSpace{N1,N2,T,N}(endog::NTuple{N1,EndogenousState{T}},
                                      exog::NTuple{N2,ExogenousState{T}},
                                      grid::Matrix{T},
                                      b::Basis{N})
    DefaultStateSpace(endog, exog, grid, grid', b)
end

function DefaultStateSpace{N1,N2,T}(endog::NTuple{N1,EndogenousState{T}},
                                    exog::NTuple{N2,ExogenousState{T}})
    grids = vcat(Vector[i.grid for i in endog], Vector[i.grid for i in exog])
    grid = gridmake(grids...)
    basis = Basis([i.basis for i in endog]..., [i.basis for i in exog]...)
    DefaultStateSpace{N1,N2,T}(endog, exog, grid, basis)
end

# function DefaultStateSpace(b::Basis, nendog::Int)
#     grid, grids = nodes(b)
#     N = ndims(B)
#     nexog = N - nendog
#     endog = tuple([EndogenousState(grids[i], b[i]) for i=1:nendog]...)
#     exog = tuple([(grids[i], b[i]) for i=nendog+1:N]...)



# function DefaultStateSpace{N1,N2,T}(endog::NTuple{N1,Vector{T}},
#                                     exog::NTuple{N2,Vector{T}})

# ------------------------------------------------------------------- #
# Frictions
# ------------------------------------------------------------------- #
include("MA_Frictions.jl")

export AbstractEntity,
       AbstractConsumer, EpsteinZinAgent, utility, is_recursive,
       AbstractProducer, CESProducer, f_k,
       AbstractVariable,
       AbstractStateVariable, EndogenousState, ExogenousState,
       AbstractExogenousProcess,
       ConstantVolatility1, ConstantVolatility, StochasticVolatility, simulate,
       AbstractStateSpace, DefaultStateSpace,
       AbstractAdjustmentCost, AdjCost, _ac, _dIac, _dkac,
       _unpack

end  # module
