"""
    intensitymap(model::AbstractModel, dims::AbstractDomain)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `IntensityMap` with `dims` an [`AbstractDomain`](@ref) as dimensions.
"""
@inline function intensitymap(
        s::M,
        dims::AbstractDomain
    ) where {M <: AbstractModel}
    return create_imgmap(intensitymap(imanalytic(M), s, dims), dims)
end
@inline intensitymap(::IsAnalytic, m::AbstractModel, dims::AbstractDomain) = intensitymap_analytic(
    m,
    dims
)
@inline intensitymap(::NotAnalytic, m::AbstractModel, dims::AbstractDomain) = intensitymap_numeric(
    m,
    dims
)

function intensitymap_analytic(s::AbstractModel, dims::AbstractDomain)
    img = allocate_imgmap(s, dims)
    intensitymap_analytic!(img, s)
    return img
end

"""
    intensitymap!(img::AbstractIntensityMap, mode;, executor = SequentialEx())

Computes the intensity map or _image_ of the `model`. This updates the `IntensityMap`
object `img`.

Optionally the user can specify the `executor` that uses `FLoops.jl` to specify how the loop is
done. By default we use the `SequentialEx` which uses a single-core to construct the image.
"""
@inline function intensitymap!(img, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end
@inline intensitymap!(::IsAnalytic, img, m::AbstractModel) = intensitymap_analytic!(img, m)
@inline intensitymap!(::NotAnalytic, img, m::AbstractModel) = intensitymap_numeric!(img, m)

function intensitymap_analytic!(img::IntensityMap, s::AbstractModel)
    return intensitymap_analytic_executor!(img, s, executor(img))
end

function intensitymap_analytic_executor!(img::IntensityMap, s::AbstractModel, ::Serial)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)
    for I in eachindex(g, bimg)
        bimg[I] = intensity_point(s, g[I]) * dx * dy
    end
    # bimg .= intensity_point.(Ref(s), g) .* dx .* dy
    return nothing
end

function intensitymap_analytic_executor!(
        img::IntensityMap, s::AbstractModel,
        ::ThreadsEx{S}
    ) where {S}
    g = domainpoints(img)
    e = executor(img)
    dx, dy = pixelsizes(img)
    @threaded e for I in CartesianIndices(g)
        img[I] = intensity_point(s, g[I]) * dx * dy
    end
    return nothing
end

function _threads_intensitymap! end
