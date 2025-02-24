export UnstructuredDomain

const DataNames = Union{<:NamedTuple{(:X, :Y, :T, :F)},<:NamedTuple{(:X, :Y, :F, :T)},
                        <:NamedTuple{(:X, :Y, :T)},<:NamedTuple{(:X, :Y, :F)},
                        <:NamedTuple{(:X, :Y)}}

# TODO make this play nice with dimensional data
struct UnstructuredDomain{D,E,H<:AMeta} <: AbstractSingleDomain{D,E}
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
function UnstructuredDomain(nt::NamedTuple; executor=Serial(), header=NoHeader())
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

function DD.rebuild(grid::UnstructuredDomain, dims, executor=executor(grid),
                    header=header(grid))
    return UnstructuredDomain(dims, executor, header)
end

Base.propertynames(g::UnstructuredDomain) = propertynames(domainpoints(g))
Base.getproperty(g::UnstructuredDomain, p::Symbol) = getproperty(domainpoints(g), p)
Base.keys(g::UnstructuredDomain) = propertynames(g)
named_dims(g::UnstructuredDomain) = StructArrays.components(dims(g))

function domainpoints(d::UnstructuredDomain)
    return getfield(d, :dims)
end

#This function helps us to lookup UnstructuredDomain at a particular Ti or Fr
#visdomain[Ti=T,Fr=F] or visdomain[Ti=T] or visdomain[Fr=F] calls work.
function Base.getindex(domain::UnstructuredDomain; Ti=nothing, Fr=nothing)
    points = domainpoints(domain)
    indices = if Ti !== nothing && Fr !== nothing
        findall(p -> (p.Ti == Ti) && (p.Fr == Fr), points)
    elseif Ti !== nothing
        findall(p -> (p.Ti == Ti), points)
    else
        findall(p -> (p.Fr == Fr), points)
    end
    return UnstructuredDomain(points[indices], executor(domain), header(domain))
end

function Base.summary(io::IO, g::UnstructuredDomain)
    n = propertynames(domainpoints(g))
    printstyled(io, "│ "; color=:light_black)
    return print(io, "UnstructuredDomain with dims: $n")
end

function Base.show(io::IO, mime::MIME"text/plain", x::UnstructuredDomain)
    println(io, "UnstructredDomain(")
    println(io, "executor: $(executor(x))")
    println(io, "Dimensions: ")
    show(io, mime, dims(x))
    return print(io, "\n)")
end

create_map(array, g::UnstructuredDomain) = UnstructuredMap(array, g)
function allocate_map(M::Type{<:AbstractArray{T}}, g::UnstructuredDomain) where {T}
    return UnstructuredMap(similar(M, size(g)), g)
end
