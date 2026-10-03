module ComradeBaseKernelAbstractionsExt

using ComradeBase
using KernelAbstractions: Backend, allocate

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}},
        g::ComradeBase.AbstractSingleDomain{<:Tuple, <:Backend},
        eldims::Tuple
    ) where {T}
    storage = allocate(executor(g), T, (size(g)..., map(length, eldims)...))
    return ComradeBase._wrapstorage(storage, g, (), Symbol(""), eldims)
end

function ComradeBase.intensitymap_analytic_executor!(
        img::ComradeBase.RectiMap,
        s::ComradeBase.AbstractModel,
        ::Backend
    )
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    f = p -> ComradeBase.intensity_point(s, p) * dx * dy
    ComradeBase._foreach_component(baseimage(img), f, Val(ndims(g))) do slab, fk
        slab .= fk.(g)
    end
    return nothing
end

function ComradeBase._pointmap!(dest, f, d, ::Backend)
    return ComradeBase._broadcast_pointmap!(dest, f, d)
end

end
