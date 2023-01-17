"""
    imagegrid(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is unallocating
since it lazily computes the grid. The grid is an example of a KeyedArray and works similarly.
This is useful for broadcasting a model across an abritrary grid.
"""
imagegrid(img::IntensityMapTypes) = grid(axiskeys(img))

imagegrid(dims::AbstractDims) = grid(dims(dims))

"""
    phasecenter(img::IntensityMap)
    phasecenter(img::StokesIntensitymap)

Computes the phase center of an intensity map. Note this is the pixels that is in
the middle of the image.
"""
function phasecenter(img::IntensityMapTypes)
    x0 = -median(img.X)
    y0 = -median(img.Y)
    return (X=x0, Y=y0)
end

"""
    imagepixels(img::IntensityMap)
    imagepixels(img::IntensityMapTypes)

Returns a named tuple with the image pixels locations `X` and `Y`.
"""
imagepixels(img::IntensityMapTypes) = (X=img.X, Y=img.Y)


function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real = 0, y0::Real = 0)
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"

    psizex=fovx/nx
    psizey=fovy/ny

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2 - x0, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2 - y0, ny)

    return (X=xitr, Y=yitr)
end


"""
    fieldofview(img::IntensityMap)
    fieldofview(img::IntensityMapTypes)

Returns a named tuple with the field of view of the image.
"""
function fieldofview(img::IntensityMapTypes)
    return fieldofview(named_axiskeys(img))
end

function fieldofview(dims::GriddedKeys)
    (;X,Y) = dims
    dx = step(X)
    dy = step(Y)
    (X=abs(last(X) - first(X))+dx, Y=abs(last(Y)-first(Y))+dy)
end


"""
    pixelsizes(img::IntensityMap)
    pixelsizes(img::IntensityMapTypes)

Returns a named tuple with the spatial pixel sizes of the image.
"""
function pixelsizes(img::IntensityMapTypes)
    keys = imagepixels(img)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end



"""
    flux(im::IntensityMap)
    flux(img::StokesIntensityMap)

Computes the flux of a intensity map
"""
function flux(im::IntensityMapTypes{T,N}) where {T,N}
    return sum(im, dims=(:X, :Y))
end

flux(im::IntensityMapTypes{T,2}) where {T} = sum(im)






"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.
"""
function centroid(im::IntensityMapTypes)
    (X, Y) = named_axiskeys(im)
    return mapslices(x->centroid(IntensityMap(x, (;X, Y))), im; dims=(:X, :Y))
end

function centroid(im::IntensityMapTypes{T,2})::Tuple{T,T} where {T}
    x0 = y0 = zero(eltype(im))
    f = flux(im)
    @inbounds for (i,x) in pairs(axiskeys(im,:X)), (j,y) in pairs(axiskeys(im,:Y))
        x0 += x.*im[X=i, Y=j]
        y0 += y.*im[X=i, Y=j]
    end
    return x0./f, y0./f
end

"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.
"""
function second_moment(im::IntensityMapTypes{T,N}) where {T,N}
    (X, Y) = named_axiskeys(im)
    return mapslices(x->second_moment(IntensityMap(x, (;X, Y))), im; dims=(:X, :Y))
end

"""
    second_moment(im::AbstractIntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.
"""
function second_moment(im::IntensityMapTypes{T,2}; center=true) where {T}
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    (;X,Y) = named_axiskeys(im)
    for (i,y) in pairs(Y), (j,x) in pairs(X)
        xx += x.^2*im[j,i]
        yy += y.^2*im[j,i]
        xy += x.*y.*im[j,i]
    end

    if center
        x0, y0 = centroid(im)
        xx = xx - x0.^2
        yy = yy - y0.^2
        xy = xy - x0.*y0
    end

    return @SMatrix [xx./f xy./f; xy./f yy./f]
end
