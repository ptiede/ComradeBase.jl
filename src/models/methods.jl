export centroid, flux, second_moment, dualmap, DualMap

function centroid(m::AbstractModel, g::AbstractDomain)
    img = intensitymap(m, g)
    return centroid(img)
end

function flux(m::AbstractModel, g::AbstractDomain)
    img = intensitymap(m, g)
    return flux(img)
end

function second_moment(m::AbstractModel, g::AbstractDomain; center = true)
    return second_moment(intensitymap(m, g); center)
end
