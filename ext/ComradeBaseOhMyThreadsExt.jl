module ComradeBaseOhMyThreadsExt

using ComradeBase
if isdefined(Base, :get_extension)
    using OhMyThreads
else
    using .OhMyThreads
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.AbstractGrid{D, <:OhMyThreads.Scheduler}) where {D}
    dx = step(dims.X)
    dy = step(dims.Y)
    g = imagegrid(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    img = tmap(g; scheduler=executor(dims)) do p
        f(p)*dx*dy
    end
    return IntensityMap(img, dims)
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,<:AbstractArray{T,N},<:ComradeBase.AbstractGrid{D, <:OhMyThreads.Scheduler}}, s::ComradeBase.AbstractModel) where {T,N,D}
    dims = axisdims(img)
    dx = step(dims.X)
    dy = step(dims.Y)
    g = imagegrid(dims)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    tforeach(CartesianIndices(img); scheduler=executor(dims)) do I
        img[I] = f(g[I])*dx*dy
    end
    return img
end



end
