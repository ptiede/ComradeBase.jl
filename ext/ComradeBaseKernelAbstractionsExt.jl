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

function ComradeBase._pointmap!(dest, f, d, ::Backend)
    return ComradeBase._broadcast_pointmap!(dest, f, d)
end

end
