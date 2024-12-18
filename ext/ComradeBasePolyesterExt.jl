module ComradeBasePolyesterExt
using ComradeBase
using Polyester: @batch

const PolyThreads = ComradeBase.ThreadsEx{:Polyester}

function ComradeBase._threads_intensitymap!(img::IntensityMap,
                                            s::ComradeBase.AbstractModel, g,
                                            ::Val{:Polyester})
    dx, dy = ComradeBase.pixelsizes(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @batch for I in CartesianIndices(pimg)
        pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function ComradeBase._threads_intensitymap!(img::UnstructuredMap,
                                            s::ComradeBase.AbstractModel, g,
                                            ::Val{:Polyester})
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @batch for I in CartesianIndices(pimg)
        pimg[I] = f(g[I])
    end
    return nothing
end

function ComradeBase._threads_visibilitymap!(vis,
                                             s::ComradeBase.AbstractModel,
                                             g,
                                             ::Val{:Polyester})
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @batch for I in CartesianIndices(g)
        pvis[I] = f(g[I])
    end
    return nothing
end

end
