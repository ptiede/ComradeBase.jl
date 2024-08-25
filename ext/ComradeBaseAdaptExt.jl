module ComradeBaseAdaptExt

using ComradeBase
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

function Adapt.adapt_structure(to, A::ComradeBase.AbstractSingleDomain)
    return rebuild(typeof(A), Adapt.adapt_structure(to, dims(A)),
                   executor(A),
                   header(A))
end

end
