export IntensityMap, SpatialIntensityMap,
    spatialgrid, pixelsizes, domainpoints,
    phasecenter, baseimage, stokes

include("intensitymap.jl")
include("coherency.jl")


export flux, centroid, second_moment, axisdims,
    spatialgrid, pixelsizes, domainpoints, phasecenter
include("methods.jl")
include("blocks.jl")

"""
    intensitymap(model::AbstractModel, dims::AbstractDomain)

Computes the intensity map or _image_ of the `model`. This returns an `IntensityMap` which
is a `IntensityMap` with `dims` an [`AbstractDomain`](@ref) as dimensions.
"""
@inline function intensitymap(
        s::M,
        dims::AbstractDomain
    ) where {M <: AbstractModel}
    return create_imgmap(_intensitymap(imanalytic(M), s, dims), dims)
end
@inline _intensitymap(::IsAnalytic, m::AbstractModel, dims::AbstractDomain) = intensitymap_analytic(
    m,
    dims
)
@inline _intensitymap(::NotAnalytic, m::AbstractModel, dims::AbstractDomain) = intensitymap_numeric(
    m,
    dims
)

function intensitymap_analytic(s::AbstractModel, dims::AbstractDomain)
    img = allocate_imgmap(s, dims)
    intensitymap_analytic!(img, s)
    return img
end

"""
    intensitymap!(img::IntensityMap, model::AbstractModel)

Computes the intensity map or _image_ of `model` in place, overwriting `img`.

The executor of the domain of `img` determines how the loop runs.
"""
@inline function intensitymap!(img, s::M) where {M}
    return _intensitymap!(imanalytic(M), img, s)
end
@inline _intensitymap!(::IsAnalytic, img, m::AbstractModel) = intensitymap_analytic!(img, m)
@inline _intensitymap!(::NotAnalytic, img, m::AbstractModel) = intensitymap_numeric!(img, m)

function intensitymap_analytic!(img::IntensityMap, s::AbstractModel)
    return intensitymap_analytic_executor!(img, s, executor(img))
end

"""
    stokes(m::AbstractArray{<:StokesParams}, p::Symbol)

Extract the specific stokes component `p` from the polarized image `m`.
"""
function stokes(m::AbstractArray{<:StokesParams}, p::Symbol)
    fm = Base.Fix2(stokes, p)
    return map(fm, m)
end

function stokes(m::StructArray{<:StokesParams}, p::Symbol)
    return getproperty(m, p)
end

stokes(m::FieldDimArray{<:StokesParams}, p::Symbol) = fieldview(m, p)

function stokes(m::StokesParams, p::Symbol)
    return getfield(m, p)
end

"""
    Returns the base image of a intensity map type object
"""
baseimage(m::AbstractArray) = m
