"""
    flux(im::IntensityMap)

Computes the flux of a intensity map
"""
function flux(im::IntensityMap{T,N}) where {T,N}
    return sum(im, dims=(:X, :Y))
end

flux(im::IntensityMap{T,2}) where {T} = sum(im)



"""
    centroid(im::IntensityMap)

Computes the image centroid aka the center of light of the image.
"""
function centroid(im::IntensityMap{T,N}) where {T<:Number,N}
    (X, Y) = named_axiskeys(im)
    return mapslices(x->centroid(IntensityMap(x, (;X, Y))), im; dims=(:X, :Y))
end

function centroid(im::IntensityMap{T,2})::Tuple{T,T} where {T<:Number}
    x0 = y0 = zero(T)
    f = flux(im)
    @inbounds for (i,x) in pairs(axiskeys(im,:X)), (j,y) in pairs(axiskeys(im,:Y))
        x0 += x.*im[X=i, Y=j]
        y0 += y.*im[X=i, Y=j]
    end
    return x0/f, y0/f
end

centroid(im::IntensityMap{<:StokesParams}) = centroid(stokes(im, :I))


"""
    second_moment(im::IntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.
"""
function second_moment(im::IntensityMap{T,N}; center=true) where {T<:Number,N}
    (X, Y) = named_axiskeys(im)
    return mapslices(x->second_moment(IntensityMap(x, (;X, Y)); center), im; dims=(:X, :Y))
end

second_moment(im::IntensityMap{<:StokesParams}; center=true) = second_moment(stokes(im, :I); center=true)

function second_moment(im::IntensityMap{T,2}; center=true) where {T<:Number}
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    (;X,Y) = named_axiskeys(im)
    for (i,y) in pairs(Y), (j,x) in pairs(X)
        xx += x^2*im[j,i]
        yy += y^2*im[j,i]
        xy += x*y*im[j,i]
    end

    if center
        x0, y0 = centroid(im)
        xx -= x0^2
        yy -= y0^2
        xy -= x0*y0
    end

    return @SMatrix [xx/f xy/f; xy/f yy/f]
end
