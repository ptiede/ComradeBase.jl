module StokedBaseAdaptExt

using StokedBase
using DimensionalData
using Adapt

function Adapt.adapt_structure(to, A::IntensityMap)
    return IntensityMap(Adapt.adapt_structure(to, DimensionalData.data(A)),
                        Adapt.adapt_structure(to, axisdims(A)),
                        Adapt.adapt_structure(to, DimensionalData.refdims(A)),
                        DimensionalData.Name(name(A)))
end

function Adapt.adapt_structure(to, A::UnstructuredMap)
    return UnstructuredMap(Adapt.adapt_structure(to, parent(A)),
                           Adapt.adapt_structure(to, axisdims(A)))
end

function Adapt.adapt_structure(to, A::StokedBase.AbstractSingleDomain)
    return rebuild(typeof(A), Adapt.adapt_structure(to, dims(A)),
                   executor(A),
                   header(A))
end

function Adapt.adapt_structure(to, A::StokedBase.LazySlice)
    return StokedBase.LazySlice(Adapt.adapt_structure(to, A.slice),
                                 A.dir,
                                 A.dims)
end

end
