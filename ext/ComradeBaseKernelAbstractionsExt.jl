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

ComradeBase._storage(b::Backend, ::Type{T}, sz) where {T} = allocate(b, T, sz)

function ComradeBase._pointmap!(img, f::F, d, ::Backend) where {F}
    return ComradeBase._broadcast_pointmap!(img, f, d)
end

function ComradeBase._convert!(f::F, basis, dest, src, ::Backend) where {F}
    return ComradeBase._broadcastconvert!(f, basis, dest, src)
end
function ComradeBase._convert!(f::F, basis, s, ::Backend) where {F}
    return ComradeBase._broadcastconvert!(f, basis, s, map(copy, s))
end

end
