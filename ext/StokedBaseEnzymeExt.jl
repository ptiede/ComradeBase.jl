module StokedBaseEnzymeExt

using StokedBase
using Enzyme: @parallel

const EnzymeThreads = StokedBase.ThreadsEx{:Enzyme}

function StokedBase._threads_intensitymap!(img::IntensityMap,
                                            s::StokedBase.AbstractModel, g,
                                            ::Val{:Enzyme})
    dx, dy = StokedBase.pixelsizes(img)
    f = Base.Fix1(StokedBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(g)
        pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function StokedBase._threads_intensitymap!(img::UnstructuredMap,
                                            s::StokedBase.AbstractModel, g,
                                            ::Val{:Enzyme})
    f = Base.Fix1(StokedBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(g)
        pimg[I] = f(g[I])
    end
    return nothing
end

function StokedBase._threads_visibilitymap!(vis,
                                             s::StokedBase.AbstractModel,
                                             g,
                                             ::Val{:Enzyme})
    f = Base.Fix1(StokedBase.visibility_point, s)
    pvis = parent(vis)
    @parallel for I in CartesianIndices(g)
        pvis[I] = f(g[I])
    end
    return nothing
end

end
