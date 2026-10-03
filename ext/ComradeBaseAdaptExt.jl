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

function Adapt.adapt_structure(to, A::ComradeBase.AbstractSingleDomain)
    return rebuild(A; dims = Adapt.adapt_structure(to, dims(A)))
end

function Adapt.adapt_structure(to, d::ComradeBase.StructuredDomain)
    return rebuild(d; coords = Adapt.adapt_structure(to, ComradeBase.coords(d)))
end

function Adapt.adapt_structure(to, A::ComradeBase.LazyGrid)
    adirs = Adapt.adapt_structure(to, A.dirs)
    return ComradeBase.LazyGrid(
        adirs,
        Adapt.adapt_structure(to, A.transform)
    )
end

function Adapt.parent_type(::Type{<:IntensityMap{T, N, D, G, A}}) where {T, N, D, G, A}
    return A
end

end
