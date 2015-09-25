module EDSUtil

using Base.Cartesian: @nloops, @nexprs

using Docile
@document

export normalize, PCA, un_transform, eds_ϵ, eds_M, eds_cheap, eval_density,
       qnwgh, n_complete, complete_inds, complete_polynomial,
       complete_polynomial!, qnwmonomial

function selectperm(v::AbstractVector,
                    k::Union{Int, OrdinalRange};
                    lt::Function=isless,
                    by::Function=identity,
                    rev::Bool=false,
                    order::Base.Order.Ordering=Base.Order.Forward)
    select!(collect(1:length(v)), k,
            Base.Order.Perm(Base.Order.ord(lt,by,rev,order),v))
end

# TODO: look into using the Distances.jl package
"""
Compute pairwise distances between a single point `x` (represented as
a d-element vector) and `n` points `y` (represented as an d × n
matrix).

Distances are stored in third argument `out`, which should be an
`n`-element vector
"""
function _pairwise!(x::Vector, y::Matrix, out::Vector)
    n_out = size(y, 2)
    nd = length(x)

    for i=1:n_out
        tmp = 0.0
        for j = 1:nd
            @inbounds tmp += (x[j] - y[j, i])^2
        end
        out[i] = tmp
    end
end

function _dist_lt_eps(D, eps)
    n = 0
    for d in D
        if d <= eps
            n += 1
        end
    end
    n
end

"""
Demean and convert to unit variance a matrix along a specified dimension

Dimension argument is optional, with default argument value `1`, which
collapses rows (i.e. the mean of each column is subtracted from the
array)
"""
normalize(x::Matrix, dim::Int=1) = (x .- mean(x, dim)) ./ std(x, dim)

"""
Normalize `x` by subtracting mean of `other` and dividing by the
standard deviation of `other`. The mean and standard deviation are computed
along dimension `dim`, which has a default value of `1`.
"""
normalize(x::Matrix, other::Matrix, dim::Int=1) =
    (x .- mean(other, dim)) ./ std(other, dim)
# NOTE: I wish I had a monad to normalize and de-normalize...
# TODO: I should instead do something like
#       `with_normalize(f::Function, x::Matrix, other::Matrix, dim::Int=1)`
#       that will normalize, call f, then denormalize.
# TODO: I could also do this with a
#       `macro with_normalization(x, other, dim, Expr)` where `Expr` is a
#       begin/end block containing the routine to be done under normalization.
#       the advantage here would be that I'm not actually using higher order
#       functions, just automating the normalization boiler-plate.

"""
The `PCA` type is used to work with the principle components of matrix of
data. The fields are

- `data`: original data
- `V`: the `V` component of the svd of the normalized `data`
- `PC`: The principal components of normalized `data`
- `PCn`: Normalized principal components
"""
immutable PCA
    data::Matrix{Float64}
    V::Matrix{Float64}
    PC::Matrix{Float64}  #  principal components
    PCn::Matrix{Float64}  #normalized principal components
end

"Construct a `PCA` object, given data"
function PCA(data::Matrix)
    datan = normalize(data)   # normalize data
    U, S, V = svd(datan)      # compute svd
    PC = datan*V              # compute principal components (PCs)
    PCn = (PC ./ std(PC, 1))  # normalize PCs
    PCA(data, V, PC, PCn)     # return object
end


"""
Given a matrix `data` and a `PCA` representation `other`, compute a
principal components representation of `data` using the `svd` of other
"""
function PCA(data::Matrix, other::PCA)
    datan = normalize(data, other.data)
    PC = datan*other.V
    PCn = (PC ./ std(other.PC, 1))
    return PCA(data, other.V, PC, PCn)
end

"""
Given `data` and `other`, construct a principal components
representation of `data`, using the svd of `other`.
"""
PCA(data::Matrix, other::Matrix) = PCA(data, PCA(other))


"""
Given `x_pcn` and a PCA representation `pca`, undo the `pca` and
normalization transformation to map `x_pcn` onto the domain of `pca.data`

Note that the `n` stands for normalized

This is typically used in this type of example:

```julia
pca = PCA(data)              # compute principal components
x_pc = do_stuff(pca)         # operate on data in PC space
x = un_transform(x_pc, pca)  # map back into domain of data
```
"""
function un_transform(x_pcn::Matrix, pca::PCA)
    x_pc = x_pcn .* std(pca.PC, 1)                     # de-normalize PCs
    x_n = x_pc * pca.V'                                 # unto PCA
    x = (x_n .* std(pca.data, 1)) .+ mean(pca.data, 1)  # de-normalize z
    x
end

"""
Constructs an epsilon distinguishable set (EDS) for a given `data` set
and minimum distance `ϵ` using algorithm P^ϵ from page 10 of MM
"""
eds_ϵ(data::Matrix{Float64}, ϵ::Float64) = eds_ϵ(PCA(data), ϵ)

"""
Constructs an epsilon distinguishable set (EDS), given the normalized
`PCA` for the `data` (obtained via `PCA(data)`) and a scalar `ϵ`
"""
function eds_ϵ(data_PCA::PCA, ϵ::Float64)
    n, d = size(data_PCA.data)

    # normalize data mean zero, unit std
    PCn = data_PCA.PCn'  # do `'` so I can operate on columns at a time

    # set up iteration objects
    EDS_PCn = Array(Float64, d, n)
    eps2 = ϵ^2
    M = 0
    N1 = n

    D_i2 = Array(Float64, N1)

    inds = Array(Int, N1)
    while N1 > 0
        M += 1
        EDS_PCn[:, M] = PCn[:, 1]  # step 3: add x_u to P^ϵ
        _pairwise!(PCn[:, 1], PCn, D_i2)  # step 1: compute D(x_i,x_j) ∀ j
        ndrop = _dist_lt_eps(D_i2, eps2)  # compute number of points to drop
        sortperm!(inds, D_i2)  # get indices that sort distances
        PCnsort = PCn[:, inds]  # sort principal components by inds
        PCn = PCnsort[:, ndrop+1:end]  # drop ones that are too close
        N1 = size(PCn, 2)  # recompte size of PCs
        D_i2 = D_i2[1:N1]  # trim distance vector
        inds = inds[1:N1]  # drop index vector
    end

    # now transpose back and undo PCA+normalization transformation
    EDS_PCn = EDS_PCn[:, 1:M]'
    EDS = un_transform(EDS_PCn, data_PCA)
    return EDS
end

"""
Given `data`, construct an EDS with a target number of points `M`.
Implements algorithm Mbar from page 13 of MM. `ϵ1` and `ϵ2` should satisfy
`ϵ1 < ϵ2` and

```julia
size(eds_ϵ(data, ϵ1), 1) > M > size(eds_ϵ(data, ϵ2), 1)
```

If no values are given, then suitable initial conditions are chosen such that:

```julia
ϵ1 = rmin / (M^(1/d) - 1)
ϵ2 = 0.5rmax * M^(1/d)
```

where `rmin` and `rmax` are the minimal and maximal norm of the normalized
principal components of `data` and `d = size(data, 2)`
"""
function eds_M(data::Matrix{Float64}, M::Int, ϵ1::Float64=NaN, ϵ2::Float64=NaN)
    d = size(data, 2)
    if any(map(isnan, [ϵ1, ϵ2]))  # find initial ϵ bounds as described in paper
        data_PCA = PCA(data)
        rmin, rmax = extrema(mapslices(norm, data_PCA.PCn, [2]))
        ϵ1 = rmin / (M^(1/d) - 1)
        ϵ2 = rmax * M^(1/d)
    else
        data_PCA = PCA(data)
    end

    ϵ1 > ϵ2  && error("ϵ1 must be smaller than ϵ2")
    # because ϵ1 < ϵ2 we will have that eds_1 has more points than eds_2
    eds_1 = eds_ϵ(data_PCA, ϵ1)
    eds_2 = eds_ϵ(data_PCA, ϵ2)

    # make sure initial guesses form a bound
    size(eds_1, 1) < M && error("ϵ1 not small enough for requested M")
    size(eds_2, 1) > M && error("ϵ2 not large enough for requested M")

    # now proceed with algorithm
    old_M_ϵ = typemax(Int)
    while true
        ϵ = (ϵ1 + ϵ2) / 2
        eds = eds_ϵ(data_PCA, ϵ)
        M_ϵ = size(eds, 1)

        # if we didn't change, stop iterating
        if M_ϵ == old_M_ϵ
            eds_1 = eds
            break
        end

        # do bisection update
        if M_ϵ > M
            ϵ1 = ϵ
        else
            ϵ2 = ϵ
        end

        # update old_M_ϵ
        old_M_ϵ = M_ϵ

    end

    return eds_1
end

"""
Construct an EDS of `data` using the cheap algorithm for computing the
essentially ergodic set. Given a set of simulated data `P`, the algorithm
proceeds as follows:

1. "Thin" the data by selecting every `κ`th point
2. Select and EDS P^ϵ of M points using `eds_ϵ` from above
3. Estimate the density at each point in P^ϵ using `eval_density`
4. Remove a fraction of points `δ` from `P` that has lowest density

Note, there is a subtlety here. We are removing points from `P`, not
`P^ϵ`. This means that each time we throw a point out of `P^ϵ`, the mass
of discarded points is not `1/size(P, 1)`, but rater the density of that
point (from step 3), over cumulative density of all `M` points in `P^ϵ`.
This means that we can toss points out of `P^ϵ` one by one until the
total mass of discarded points is equal to `δ`.

**NOTE**: In this case `data` is assumed to be the raw data obtained via
simulation, not the essentially ergodic set (as is assumed in the other
`eds_*` methods). This routine will optionally select every `κ`th point,
but this argument has a default value of `1`
"""
function eds_cheap(data::Matrix{Float64}, ϵ::Float64, δ::Float64, κ::Int=1)
    # TODO: implement this
    nothing
end

"""
TODO: write docstring
"""
function eds_locally_adaptive(data::Matrix{Float64}, ϵ::Vector{Float64})
    n, d = size(data)
    @assert n == length(ϵ) "length(ϵ) must be same as size(data, 1)"

    # TODO: implement this algorithm. It might take some API work because
    #       we need information on function approximation errors at each point
    nothing
end


# Somehow the apple libm is faster than the one julia calls on OSX
@osx? (
         begin
             myexp(x::Float64) = ccall((:exp, :libm), Float64, (Float64,), x)
         end
       : begin
             myexp(x::Float64) = exp(x)
         end
       )

"""
Estimate the density function in a given set of `points` by applying
a multivariate kernel algorithm to the `data`. See formula (2) in MM.

First argument is the data points, with `size(⋅) = (n, d)`

Second argument is the points at which to evaluate the density, with
`size(⋅) = (npoints, d)`

Output is three items:

1. `density`: Estimated density function with `size(⋅) = (npoints,)`
2. `points_PCn`: normalized principal components of the matrix of
   data for which the density was estimated with `size(⋅)=(npoints, d)`.
   This is needed by the EDS algorithm so it is returned here to reduce
   duplicate computation
3. `di_min`: The distance from each point in `data` to the closest
   neighbor with `size(⋅)=(npoints,)`. Also needed by the EDS
   algorithm so it is returned here to reduce duplicate computation
"""
function eval_density(data::Matrix, points::Matrix)

    n, d = size(data)

    n_points = size(points, 1)

    # transform data
    data_PCA = PCA(data)
    PC = data_PCA.PC
    PCn = data_PCA.PCn'  # do `'` so I can operate on columns at a time

    # transform points
    points_PCA = PCA(points, data_PCA)
    pts_PCn = points_PCA.PCn'  # same comment about `'`

    # Define constants
    hbar = n^(-1.0/(d+4.0))
    constant = 1/(n*hbar^(d) * (2π)^(d/2))
    constant2 = -0.5/hbar^2

    # allocate space
    density = Array(Float64, n_points)
    Di_min = Array(Float64, n_points)

    # apply formula (2)
    for i=1:n_points  # loop over all points
        dens_i = 0.0
        min_di2 = Inf
        for j=1:n_points  # loop over all other points
            d_i2_j = 0.0
            for k=1:d  # loop over d
                @inbounds d_i2_j += (pts_PCn[k, i] - PCn[k, j])^2
            end
            dens_i += myexp(d_i2_j*constant2)
            if i != j && d_i2_j < min_di2
                min_di2 = d_i2_j
            end
        end
        density[i] = dens_i
        Di_min[i] = sqrt(min_di2)
    end

    density .*= constant


    return density, points_PCA.PCn, Di_min
end

"""
WARNING: this function doesn't do anything yet...

Constructs `n` clusters and computes clusters' centers for a given
`data` set.

See Judd, Maliar and Maliar, (2010),
"A Cluster-Grid Projection Method: Solving Problems with High Dimensionality",
NBER Working Paper 15965 (henceforth, JMM, 2010).
"""
function find_clusters(data::Matrix, n::Int)
    T, M = size(data)
    data_PCA = PCA(data)
    PCn = data_PCA.PCn'  # `'` so I can operate on columns at a time

    # TODO: don't have `clusterdata` function
end


# --------------------------------------------------------- #
# Stuff to construct basis matrices of complete polynomials #
# --------------------------------------------------------- #

"""
Construct basis matrix for complete polynomial of degree `d`, given
input data `z`. `z` is assumed to be the degree 1 realization of each
variable. For example, if variables are `q`, `r`, and `s`, then `z`
should be `z = [q r s]`

Output is a basis matrix. In our example, with `d` set to 2 we would have

TODO: update docstring to properly give order of terms

```julia
out = [ones(size(z,1)) q r s q.*r q.*s r.*s q.^2 q.*r q.*s r.^2 r.*s s.^2]
```
"""
:complete_polynomial

immutable Degree{N} end

function n_complete(n::Int, D::Int)
    out = 1
    for d=1:D
        tmp = 1
        for j=0:d-1
            tmp *= (n+j)
        end
        out += div(tmp, factorial(d))
    end
    out
end

@generated function complete_polynomial!{N}(z::Matrix, d::Degree{N},
                                            out::Matrix)
    complete_polynomial_impl!(z, d, out)
end

function complete_polynomial_impl!{T,N}(z::Type{Matrix{T}}, ::Type{Degree{N}},
                                        ::Type{Matrix{T}})
    quote
        nobs, nvar = size(z)
        if size(out) != (nobs, n_complete(nvar, $N))
            error("z, out not compatible")
        end

        # reset first column to ones
        @inbounds for i=1:nobs
            out[i, 1] = 1.0
        end

        # set next nvar columns to input matrix
        @inbounds for n=2:nvar+1, i=1:nobs
            out[i, n] = z[i, n-1]
        end

        ix = nvar+1
        @nloops($N, # number of loops
                i,  # counter
                d->((d == $N ? 1 : i_{d+1}) : nvar),  # ranges
                d->(d == $N ? nothing :
                    (begin
                        ix += 1
                        for r=1:nobs
                            tmp = one($T)
                            @nexprs $N-d+1 j->(tmp *= z[r, i_{$N-j+1}])
                            out[r, ix]=tmp
                        end
                    end)),  # preexpr
                Expr(:block, :nothing)  # bodyexpr
                )
        out
    end
end

function complete_polynomial{T}(z::Matrix{T}, d::Int)
    nobs, nvar = size(z)
    out = Array(T, nobs, n_complete(nvar, d))
    complete_polynomial!(z, Degree{d}(), out)::Matrix{T}
end

function complete_polynomial!{T}(z::Matrix{T}, d::Int, out::Matrix{T})
    complete_polynomial!(z, Degree{d}(), out)::Matrix{T}
end

# ------------------------------------------------- #
# Routines to generate quadrature weights and nodes #
# ------------------------------------------------- #

"""
Gauss Hermite quadrature notes and weights in N dimensions. Limited to
no more than 10 nodes in each dimension.

TODO: I really don't like this. I'll probably swap it out for one of my
      CompEcon routines once I have verified that my code gives the
      same answer as the Matlab
"""
function qnwgh(n::Int=10, d::Int=1, vcv::Matrix=eye(d))
    if n == 1
        ϵ = [0.0]
        ω = [sqrt(pi)]
    elseif n == 2
        ϵ = [0.7071067811865475, -0.7071067811865475]
        ω = [0.8862269254527580, 0.8862269254527580]
    elseif n == 3
        ϵ = [1.224744871391589, 0, -1.224744871391589]
        ω = [0.2954089751509193, 1.181635900603677, 0.2954089751509193]
    elseif n == 4
        ϵ = [1.650680123885785, 0.5246476232752903,  -0.5246476232752903,
             -1.650680123885785]
        ω = [0.08131283544724518, 0.8049140900055128, 0.8049140900055128,
             0.08131283544724518]
    elseif n == 5
        ϵ = [2.020182870456086, 0.9585724646138185, 0, -0.9585724646138185,
             -2.020182870456086]
        ω = [0.01995324205904591, 0.3936193231522412, 0.9453087204829419,
             0.3936193231522412, 0.01995324205904591]
    elseif n == 6
        ϵ = [2.350604973674492, 1.335849074013697, 0.4360774119276165,
             -0.4360774119276165, -1.335849074013697, -2.350604973674492]
        ω = [0.004530009905508846, 0.1570673203228566, 0.7246295952243925,
             0.7246295952243925, 0.1570673203228566, 0.004530009905508846]
    elseif n == 7
        ϵ = [2.651961356835233, 1.673551628767471, 0.8162878828589647, 0,
             -0.8162878828589647, -1.673551628767471, -2.651961356835233]
        ω = [0.0009717812450995192, 0.05451558281912703, 0.4256072526101278,
             0.8102646175568073, 0.4256072526101278, 0.05451558281912703,
             0.0009717812450995192]
    elseif n == 8
        ϵ = [2.930637420257244, 1.981656756695843, 1.157193712446780,
             0.3811869902073221, -0.3811869902073221, -1.157193712446780,
             -1.981656756695843,-2.930637420257244]
        ω = [0.0001996040722113676, 0.01707798300741348, 0.2078023258148919,
             0.6611470125582413, 0.6611470125582413, 0.2078023258148919,
             0.01707798300741348, 0.0001996040722113676]
    elseif n == 9
        ϵ = [3.190993201781528, 2.266580584531843, 1.468553289216668,
             0.7235510187528376, 0,-0.7235510187528376, -1.468553289216668,
             -2.266580584531843, -3.190993201781528]
        ω = [0.00003960697726326438, 0.004943624275536947,
             0.08847452739437657, 0.4326515590025558, 0.7202352156060510,
             0.4326515590025558, 0.08847452739437657, 0.004943624275536947,
             0.00003960697726326438]
    elseif n == 10
        ϵ = [3.436159118837738, 2.532731674232790, 1.756683649299882,
             1.036610829789514, 0.3429013272237046, -0.3429013272237046,
             -1.036610829789514, -1.756683649299882, -2.532731674232790,
             -3.436159118837738]
        ω = [7.640432855232621e-06, 0.001343645746781233,
             0.03387439445548106, 0.2401386110823147, 0.6108626337353258,
             0.6108626337353258, 0.2401386110823147, 0.03387439445548106,
             0.001343645746781233, 7.640432855232621e-06]
    else
        error("n must be between 1 and 10")
    end

    n_nodes = n^d
    z1 = zeros(n_nodes, d)
    ω1 = ones(n_nodes)

    for i=1:d
        ix = 1
        for j=1:n^(d-i)
            for u=1:n
                n_new = n^(i-1)
                z1[ix:ix+n_new-1, i] = ϵ[u]
                ω1[ix:ix+n_new-1] .*= ω[u]
                ix += n_new
            end
        end
    end

    z = sqrt(2) .* z1
    weights = ω1 ./ (sqrt(π)^d)
    nodes = ifelse(length(vcv) == 1, z*chol(vcv)[1], z*chol(vcv))::Matrix{Float64}
    return nodes, weights
end

"""
Computes integration nodes and weights under an N-dimensional monomial
(non-product) integration rule. If `kind` is equal to `:first` (the
default), then `2n` nodes will be computed, otherwise an algorithm
producing `2n^2+1` nodes is used.
"""
:qnwmonomial

qnwmonomial(n::Int, vcv::Matrix{Float64}, kind::Symbol=:first) =
    kind == :first ? _qnwmonomial1(n, vcv) : _qnwmonomial2(n, vcv)


function _qnwmonomial1(n::Int, vcv::Matrix{Float64})
    n_nodes = 2n

    z1 = zeros(n_nodes, n)

    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i=1:n
        z1[2*(i-1)+1:2*i, i] = [1, -1]
    end

    sqrt_vcv = chol(vcv)
    R = sqrt(n)*sqrt_vcv
    ϵj = z1*R
    ωj = ones(n_nodes) ./ n_nodes
    ϵj, ωj
end


function _qnwmonomial2(n::Int, vcv::Matrix{Float64})
    n_nodes = 2n^2 + 1
    z0 = zeros(1, n)

    z1 = zeros(2n, n)
    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i=1:n
        z1[2*(i-1)+1:2*i, i] = [1, -1]
    end

    z2 = zeros(2n*(n-1), n)
    i = 0

    # In each node, a pair of random variables (p,q) takes either values
    # (1,1) or (1,-1) or (-1,1) or (-1,-1), and all other variables take
    # value 0. For example, for N = 2, `z2 = [1 1; 1 -1; -1 1; -1 1]`
    for p=1:n-1
        for q=p+1:n
            i += 1
            z2[4*(i-1)+1:4*i, p] = [1, -1, 1, -1]
            z2[4*(i-1)+1:4*i, q] = [1, 1, -1, -1]
        end
    end

    sqrt_vcv = chol(vcv)
    R = sqrt(n+2)*sqrt_vcv
    S = sqrt((n+2)/2)*sqrt_vcv
    ϵj = [z0; z1*R; z2*S]
    ωj = vcat(2/(n+2) * ones(size(z0, 1)),
              (4-n)/(2*(n+2)^2) * ones(size(z1, 1)),
               1/(n+2)^2 * ones(size(z2, 1)))
    return ϵj, ωj
end

end  # module
