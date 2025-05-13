module ComradeBaseEnzymeExt

using ComradeBase
using Enzyme: @parallel

const EnzymeThreads = ComradeBase.ThreadsEx{:Enzyme}

function ComradeBase._threads_intensitymap!(
        img::IntensityMap,
        s::ComradeBase.AbstractModel, g,
        ::Val{:Enzyme}
    )
    dx, dy = ComradeBase.pixelsizes(img)
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(g)
        pimg[I] = f(g[I]) * dx * dy
    end
    return nothing
end

function ComradeBase._threads_intensitymap!(
        img::UnstructuredMap,
        s::ComradeBase.AbstractModel, g,
        ::Val{:Enzyme}
    )
    f = Base.Fix1(ComradeBase.intensity_point, s)
    pimg = parent(img)
    @parallel for I in CartesianIndices(g)
        pimg[I] = f(g[I])
    end
    return nothing
end

function ComradeBase._threads_visibilitymap!(
        vis,
        s::ComradeBase.AbstractModel,
        g,
        ::Val{:Enzyme}
    )
    f = Base.Fix1(ComradeBase.visibility_point, s)
    pvis = parent(vis)
    @parallel for I in CartesianIndices(g)
        pvis[I] = f(g[I])
    end
    return nothing
end

end
