module ComradeBaseKernelAbstractionsExt

using ComradeBase
using KernelAbstractions: Backend, allocate
using StructArrays

function ComradeBase.allocate_map(::Type{<:AbstractArray{T}}, g::UnstructuredDomain{D, <: Backend}) where {T, D}
    return ComradeBase.UnstructuredMap(allocate(executor(g), T, size(g)), g)
end

function ComradeBase.allocate_map(::Type{<:StructArray{T}}, g::UnstructuredDomain{D, <: Backend}) where {T, D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x->allocate(exec, x, size(g)), T)
    return UnstructuredMap(arrs, g)
end


function allocate_map(::Type{<:AbstractArray{T}}, g::ComradeBase.AbstractRectiGrid{D, <: Backend}) where {T, D}
    executor = executor(g)
    return IntensityMap(allocate(executor, T, size(g)), g)
end

function allocate_map(::Type{<:StructArray{T}}, g::ComradeBase.AbstractRectiGrid{D, <: Backend}) where {T, D}
    exec = executor(g)
    arrs = StructArrays.buildfromschema(x->allocate(exec, x, size(g)), T)
    return IntensityMap(arrs, g)
end



end
