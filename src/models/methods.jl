export centroid, flux, second_moment, dualmap

function centroid(m::AbstractModel, g::AbstractDomain)
    img = intensitymap(m, g)
    return centroid(img)
end

function flux(m::AbstractModel, g::AbstractDomain)
    img = intensitymap(m, g)
    return flux(img)
end

function second_moment(m::AbstractModel, g::AbstractDomain)
    img = intensitymap(m, g)
    return second_moment(img)
end

"""
    dualmap(m::AbstractModel, dims::AbstractDualDomain)

Computes both the intensity map and visibility map of the `model`. This can be faster
than computing them separately as some intermediate results can be reused.
This returns a tuple `(img, vis)` where `img` is the intensity map and `vis` the visibility map.
"""
function dualmap(m::AbstractModel, dims::AbstractDualDomain)
    return intensitymap(m, dims), visibilitymap(m, dims)
end

