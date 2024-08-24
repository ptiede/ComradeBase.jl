module ComradeBaseAdaptExt

using ComradeBase
using DimensionalData
using Adapt

function Adapt.adapt_structure(to, A::IntensityMap)
    rebuild(A, 
        Adapt.adapt_structure(to, parent(A)),
        Adapt.adapt_structure(to, dims(A)),
        Adapt.adapt_structure(to, refdims(A)),
        DimensionalData.Name(name(A)),
        Adapt.adapt_structure(to, metadata(A))
    )
end

function Adapt.adapt_structure(to, A::UnstructuredMap)
    return UnstructuredMap(Adapt.adapt_structure(to, parent(A)),
                           Adapt.adapt_structure(to, axisdims(A)))
end

function Adapt.adapt_structure(to, A::AbstractSingleDomain)
    return UnstructuredDomain(Adapt.adapt_structure(to, dims(A)), 
                              Adapt.adapt_structure(executor(A)), 
                              Adapt.adapt_structure(header(A))
                              )
end

end
