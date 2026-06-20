export UnstructuredDomain, regroup

const DataNames = Union{
    <:NamedTuple{(:X, :Y, :T, :F)}, <:NamedTuple{(:X, :Y, :F, :T)},
    <:NamedTuple{(:X, :Y, :T)}, <:NamedTuple{(:X, :Y, :F)},
    <:NamedTuple{(:X, :Y)},
}

# TODO make this play nice with dimensional data
struct UnstructuredDomain{D, E, H <: AMeta} <: AbstractSingleDomain{D, E}
    dims::D
    executor::E
    header::H
end

EnzymeRules.inactive_type(::Type{<:UnstructuredDomain}) = true

"""
    UnstructuredDomain(dims::NamedTuple; executor=Serial(), header=ComradeBase.NoHeader)

Builds an unstructured grid (really a vector of points) from the dimensions `dims`.
The `executor` is used controls how the grid is computed when calling
`visibilitymap` or `intensitymap`. The default is `Serial` which mean regular CPU computations.
For threaded execution use [`ThreadsEx()`](@ref) or load `OhMyThreads.jl` to uses their schedulers.

Note that unlike `RectiGrid` which assigns dimensions to the grid points, `UnstructuredDomain`
does not. This is becuase the grid is unstructured the points are a cloud in a space
"""
function UnstructuredDomain(nt::NamedTuple; executor = Serial(), header = NoHeader())
    p = StructArray(nt)
    return UnstructuredDomain(p, executor, header)
end

Base.ndims(d::UnstructuredDomain) = ndims(dims(d))
Base.size(d::UnstructuredDomain) = size(dims(d))
Base.firstindex(d::UnstructuredDomain) = firstindex(dims(d))
Base.lastindex(d::UnstructuredDomain) = lastindex(dims(d))
#Make sure we actually get a tuple here
# Base.front(d::UnstructuredDomain) = UnstructuredDomain(Base.front(StructArrays.components(dims(d))), executor=executor(d), header=header(d))
# Base.eltype(d::UnstructuredDomain) = Base.eltype(dims(d))

function DD.rebuild(
        grid::UnstructuredDomain, dims, executor = executor(grid),
        header = header(grid)
    )
    return UnstructuredDomain(dims, executor, header)
end

function DD.rebuild(
        grid::UnstructuredDomain; dims = dims(grid), executor = executor(grid),
        header = header(grid)
    )
    return rebuild(grid, dims, executor, header)
end

Base.propertynames(g::UnstructuredDomain) = propertynames(domainpoints(g))
@inline Base.getproperty(g::UnstructuredDomain, p::Symbol) = getproperty(domainpoints(g), p)
Base.keys(g::UnstructuredDomain) = propertynames(g)
named_dims(g::UnstructuredDomain) = StructArrays.components(dims(g))

function domainpoints(d::UnstructuredDomain)
    return getfield(d, :dims)
end

#This function helps us to lookup UnstructuredDomain at a particular Ti or Fr
#visdomain[Ti=T,Fr=F] or visdomain[Ti=T] or visdomain[Fr=F] calls work.
function Base.getindex(domain::UnstructuredDomain; Ti = nothing, Fr = nothing)
    points = domainpoints(domain)
    indices = if Ti !== nothing && Fr !== nothing
        findall(p -> (p.Ti == Ti) && (p.Fr == Fr), points)
    elseif Ti !== nothing
        findall(p -> (p.Ti == Ti), points)
    elseif Fr !== nothing
        findall(p -> (p.Fr == Fr), points)
    end
    return UnstructuredDomain(points[indices], executor(domain), header(domain))
end

"""
    regroup(domain::UnstructuredDomain, axes::Symbol...)

Return `(rdomain, perm)` where `rdomain` is `domain` reordered by a stable **lexicographic** sort over
the points' `axes` properties so that all points sharing the same `axes` values are contiguous, and
`perm` is the permutation such that `domainpoints(rdomain) == domainpoints(domain)[perm]`.

With a single axis (`regroup(d, :Fr)`) this groups by frequency; with several
(`regroup(d, :Fr, :Ti)`) the first axis is the major key, so the data nests as `Fr`-blocks each split
into `Ti`-blocks — matching a sharding declared as `ReactantEx(mesh; Fr = :devf, Ti = :devt)` (the
declaration's axis order sets the same major/minor order). This is the companion to sharding the
corresponding image dimensions across a device mesh (see [`shard_frequency`](@ref)/[`shard_time`](@ref)):
contiguous-per-group ordering lets the per-block visibilities lay out cleanly across devices. Apply
`perm` to any data/noise vectors that must stay aligned with the model visibilities, and use
`invperm(perm)` to map results back to the original order.

!!! note
    Even per-device placement of whole groups requires the groups to be (close to) equal sized. With
    ragged group sizes the grouping is still contiguous but a uniform mesh split will not fall exactly
    on group boundaries.
"""
function regroup(domain::UnstructuredDomain, axes::Symbol...)
    isempty(axes) && throw(ArgumentError("regroup requires at least one axis"))
    points = domainpoints(domain)
    cols = map(a -> getproperty(points, a), axes)
    keyvecs = collect(zip(cols...))   # vector of tuples, compared lexicographically
    perm = sortperm(keyvecs; alg = Base.Sort.DEFAULT_STABLE)
    rdomain = UnstructuredDomain(points[perm], executor(domain), header(domain))
    return rdomain, perm
end

function Base.summary(io::IO, g::UnstructuredDomain)
    n = propertynames(domainpoints(g))
    printstyled(io, "│ "; color = :light_black)
    return print(io, "UnstructuredDomain with dims: $n")
end

function Base.show(io::IO, mime::MIME"text/plain", x::UnstructuredDomain)
    println(io, "UnstructuredDomain(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    return print(io, "\n)")
end

create_map(array, g::UnstructuredDomain) = UnstructuredMap(array, g)
function allocate_map(M::Type{<:AbstractArray{T}}, g::UnstructuredDomain) where {T}
    return UnstructuredMap(similar(M, size(g)), g)
end
