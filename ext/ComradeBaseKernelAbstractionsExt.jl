module ComradeBaseKernelAbstractionsExt

using ComradeBase
using KernelAbstractions: Backend, allocate

function ComradeBase.allocate_map(
        ::Type{<:AbstractArray{T}}, g::ComradeBase.AbstractSingleDomain{<:Tuple, <:Backend}
    ) where {T}
    return IntensityMap(allocate(executor(g), T, size(g)), g)
end

ComradeBase._storage(b::Backend, ::Type{T}, sz) where {T} = allocate(b, T, sz)

function ComradeBase._pointmap!(img, f::F, d, ::Backend) where {F}
    return ComradeBase._broadcast_pointmap!(img, f, d)
end

end
