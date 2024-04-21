export IntensityMap, SpatialIntensityMap,
        DataArr, SpatialDataArr, DataNames,
        named_axisdims, imagepixels, pixelsizes, domainpoints,
        phasecenter, baseimage

include("dim_image.jl")
include("stokes_image.jl")
const IntensityMapTypes{T,N} = Union{IntensityMap{T,N}, StokesIntensityMap{T,N}}

export flux, centroid, second_moment, named_axisdims, axisdims,
       imagepixels, pixelsizes, domainpoints, phasecenter
include("methods.jl")
include("io.jl")


"""
    intensitymap(model::AbstractModel, dims::AbstractDomain)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `IntensityMap` with `dims` an [`AbstractDomain`](@ref) as dimensions.
"""
@inline function intensitymap(s::M,
                              dims::AbstractDomain
                              ) where {M<:AbstractModel}
    return create_map(intensitymap(imanalytic(M), s, dims), dims)
end
@inline intensitymap(::IsAnalytic, m::AbstractModel, dims::AbstractDomain)  = intensitymap_analytic(m, dims)
@inline intensitymap(::NotAnalytic, m::AbstractModel, dims::AbstractDomain) = intensitymap_numeric(m, dims)


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
@inline intensitymap!(::IsAnalytic, img, m::AbstractModel)  = intensitymap_analytic!(img, m)
@inline intensitymap!(::NotAnalytic, img, m::AbstractModel) = intensitymap_numeric!(img, m)




export StokesIntensityMap, stokes
