module ComradeBaseEnzymeExt

using ComradeBase
using Enzyme: @parallel

const EnzymeThreads = ComradeBase.ThreadsEx{:Enzyme}

function ComradeBase.intensitymap_analytic_executor!(
                                            img::IntensityMap,
                                            s::ComradeBase.AbstractModel, 
                                            ::EnzymeThreads)
    dx, dy = ComradeBase.pixelsizes(img)
    g = ComradeBase.domainpoints(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(pimg)
        pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function ComradeBase.intensitymap_analytic_executor!(img::UnstructuredMap,
                                            s::ComradeBase.AbstractModel,
                                            ::EnzymeThreads)
    g = ComradeBase.domainpoints(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(pimg)
        pimg[I] = f(g[I])
    end
    return nothing
end

function ComradeBase.visibilitymap_analytic_executor!(vis::ComradeBase.FluxMap2,
                                             s::ComradeBase.AbstractModel,
                                             ::EnzymeThreads)
    dims = axisdims(vis)
    g = domainpoints(dims)
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @parallel for I in CartesianIndices(g)
        pvis[I] = f(g[I])
    end
    return nothing
end

end
