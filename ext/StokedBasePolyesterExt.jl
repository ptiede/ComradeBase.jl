module StokedBasePolyesterExt
using StokedBase
using Polyester: @batch

const PolyThreads = StokedBase.ThreadsEx{:Polyester}

function StokedBase._threads_intensitymap!(img::IntensityMap,
                                           s::StokedBase.AbstractModel, g,
                                           ::Val{:Polyester})
    dx, dy = StokedBase.pixelsizes(img)
    f = Base.Fix1(StokedBase.intensity_point, s)
    pimg = parent(img)
    @batch for I in eachindex(pimg)
        @inbounds pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function StokedBase._threads_intensitymap!(img::UnstructuredMap,
                                           s::StokedBase.AbstractModel, g,
                                           ::Val{:Polyester})
    f = Base.Fix1(StokedBase.intensity_point, s)
    pimg = parent(img)
    @batch for I in eachindex(pimg)
        @inbounds pimg[I] = f(g[I])
    end
    return nothing
end

function StokedBase._threads_visibilitymap!(vis,
                                            s::StokedBase.AbstractModel,
                                            g,
                                            ::Val{:Polyester})
    f = Base.Fix1(StokedBase.visibility_point, s)
    pvis = parent(vis)
    @batch for I in eachindex(pvis)
        @inbounds pvis[I] = f(g[I])
    end
    return nothing
end

end
