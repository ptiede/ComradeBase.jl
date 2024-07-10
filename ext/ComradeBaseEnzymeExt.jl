module ComradeBaseEnzymeExt

using ComradeBase
using Enzyme: @parallel

const EnzymeThreads = ComradeBase.ThreadsEx{:enzyme}

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.AbstractRectiGrid{D, <:EnzymeThreads}) where {D}
    img = ComradeBase.allocate_imgmap(s, dims)
    ComradeBase.intensitymap_analytic!(img, s)
    return img
end

function ComradeBase.intensitymap_analytic(s::ComradeBase.AbstractModel, dims::ComradeBase.UnstructuredDomain{D, <:EnzymeThreads}) where {D}
    img = ComradeBase.allocate_imgmap(s, dims)
    ComradeBase.intensitymap_analytic!(img, s)
    return img
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T, N, D, <:ComradeBase.AbstractRectiGrid{D, <:EnzymeThreads}}, s::ComradeBase.AbstractModel) where {T, N, D}
    dx, dy = ComradeBase.pixelsizes(img)
    g = ComradeBase.domainpoints(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    I = CartesianIndices(img)
    @parallel for I in CartesianIndices(img)
        img[I] = f(g[I])*dx*dy
    end
    return nothing
end

function ComradeBase.intensitymap_analytic!(img::UnstructuredMap{T, N, <:ComradeBase.UnstructuredDomain{D, <:EnzymeThreads}}, s::ComradeBase.AbstractModel) where {T, N, D}
    g = ComradeBase.domainpoints(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    I = CartesianIndices(img)
    @parallel for I in CartesianIndices(img)
        img[I] = f(g[I])
    end
    return nothing
end

end
