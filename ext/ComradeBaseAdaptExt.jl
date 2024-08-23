module ComradeBaseAdaptExt

using ComradeBase
using Adapt

function Adapt.adapt_structure(to, A::UnstructuredMap)
    return UnstructuredMap(Adapt.adapt_structure(to, parent(A)),
                           Adapt.adapt_structure(to, axisdims(A)))
end

function Adapt.adapt_structure(to, A::UnstructuredDomain)
    return UnstructuredDomain(Adapt.adapt_structure(to, dims(A)), executor(A), header(A))
end

function Adapt.adapt_structure(to, A::RectiGrid)
    return RectiGrid(Adapt.adapt_structure(to, dims(A)), executor(A), header(A))
end

end
