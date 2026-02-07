module ComradeBaseAdaptExt

using ComradeBase
using DimensionalData
using Adapt

function Adapt.adapt_structure(to, A::IntensityMap)
    return IntensityMap(
        Adapt.adapt_structure(to, DimensionalData.data(A)),
        Adapt.adapt_structure(to, axisdims(A)),
        Adapt.adapt_structure(to, DimensionalData.refdims(A)),
        DimensionalData.Name(name(A))
    )
end

function Adapt.adapt_structure(to, A::UnstructuredMap)
    return UnstructuredMap(
        Adapt.adapt_structure(to, parent(A)),
        Adapt.adapt_structure(to, axisdims(A))
    )
end

function Adapt.adapt_structure(to, A::ComradeBase.AbstractSingleDomain)
    return rebuild(A; dims = Adapt.adapt_structure(to, dims(A)))
end

function Adapt.adapt_structure(to, A::ComradeBase.LazySlice)
    return ComradeBase.LazySlice(
        Adapt.adapt_structure(to, A.slice),
        A.dir,
        A.dims
    )
end

function Adapt.parent_type(::Type{<:IntensityMap{T, N, D, G, A}}) where {T, N, D, G, A}
    return A
end

function Adapt.parent_type(::Type{<:UnstructuredMap{T, A}}) where {T, A}
    return A
end

end
