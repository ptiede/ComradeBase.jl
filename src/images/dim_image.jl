using DimensionalData
const DD = DimensionalData
using DimensionalData: AbstractDimArray, NoName, NoMetadata, format


DD.@dim Fr "frequency"


struct DimIntensityMap{T,N,D<:Tuple,R<:Tuple,A<:AbstractArray{T,N},Na,Me} <: AbstractDimArray{T,N,D,A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me
end

DimIntensityMap(data::AbstractArray, dims; kwargs...) = DimIntensityMap(data, (dims,); kw...)

function DimIntensityMap(
        data::AbstractArray, dims::Union{Tuple, NamedTuple};
        refdims=(), name=NoName(), metadata=NoMetadata())

    return DimIntensityMap(data, format(dims, data), refdims, name, metadata)
end

function DimIntensityMap(; data, dims, refdims=(), name=NoName(), metadata=NoMetadata())
    return DimIntensityMap(data, dims; refdims, name, metadata)
end


function DimIntensityMap(A::AbstractDimArray;
    data=data(A), dims=dims(A), refdims=refdims(A), name=name(A), metadata=metadata(A)
)
    return DimIntensityMap(data, dims; refdims, name, metadata)
end

@inline function DD.rebuild(
        ::DimIntensityMap, data::AbstractArray,
        dims::Tuple, refdims::Tuple, name, metadata)

    return DimIntensityMap(data, dims, refdims, name, metadata)
end

function StokesDimIntensityMap(I::DimIntensityMap, Q::DimIntensityMap, U::DimIntensityMap, V::DimIntensityMap)
    pI = parent(I)
    pQ = parent(Q)
    pU = parent(U)
    pV = parent(V)
    return rebuild(I, StructArray{StokesParams{eltype(pI)}}((I=pI, Q=pQ, U=pU, V=pV)))
end
