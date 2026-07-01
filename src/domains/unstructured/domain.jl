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
# Shared core for every `regroup` method: build one stable **lexicographic** permutation from the
# per-point key columns (first column is the major key) and return the reordered domain plus `perm`.
# All flavours of grouping (by raw value, or by grid-plane position) differ only in how they build
# `keycols`, so the ordering lives in exactly one place.
function _regroup(domain::UnstructuredDomain, keycols)
    points = domainpoints(domain)
    keyvecs = collect(zip(keycols...))   # vector of tuples, compared lexicographically
    perm = sortperm(keyvecs; alg = Base.Sort.DEFAULT_STABLE)
    rdomain = UnstructuredDomain(points[perm], executor(domain), header(domain))
    return rdomain, perm
end

function regroup(domain::UnstructuredDomain, axes::Symbol...)
    isempty(axes) && throw(ArgumentError("regroup requires at least one axis"))
    points = domainpoints(domain)
    return _regroup(domain, map(a -> getproperty(points, a), axes))
end

# Rank a coordinate `v` by its position within the grid axis values `order`, matched with `isapprox`
# so a data frequency/time that differs from the grid's by floating-point round-off still lands on the
# right plane. A coordinate absent from the axis sorts to the end (`lastindex + 1`).
_gridrank(order, v) = something(findfirst(x -> isapprox(x, v), order), lastindex(order) + 1)

"""
    regroup(domain::UnstructuredDomain, grid::AbstractRectiGrid)

Reorder `domain` into the plane order the multidomain Fourier transform enumerates the image planes
of `grid` — `DimPoints(dims(grid)[3:end])`, i.e. column-major over the grid's *stored* non-spatial
(`Ti`/`Fr`) coordinate order. Unlike the value-sorting [`regroup`](@ref)`(domain, axes...)`, points are
ranked by their *position within* each grid axis (matched with `isapprox`), so a descending or
non-monotonic grid still lays each block out to co-locate with its image plane — the precondition for
sharding the blocks across a device mesh. Points whose coordinate is absent from a grid axis sort to
the end. Returns `(rdomain, perm)`.

This is the single source of the data-side plane ordering; it derives its axes from the same
`dims(grid)[3:end]` the NUFFT planner iterates, so the two cannot silently drift apart.
"""
function regroup(domain::UnstructuredDomain, grid::AbstractRectiGrid)
    points = domainpoints(domain)
    # non-spatial image axes in stored (column-major) order, keeping only those the data carries
    gaxes = filter(a -> hasproperty(points, a), map(name, dims(grid)[3:end]))
    isempty(gaxes) && return domain, Base.OneTo(length(points))
    ranks = map(gaxes) do a
        order = getproperty(grid, a)
        map(v -> _gridrank(order, v), getproperty(points, a))
    end
    # `DimPoints` varies the first non-spatial dim fastest, so the last grid dim is the major sort key.
    return _regroup(domain, reverse(ranks))
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
