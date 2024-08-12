"""
    domainpoints(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is unallocating
since it lazily computes the grid. The grid is an example of a DimArray and works similarly.
This is useful for broadcasting a model across an abritrary grid.
"""
domainpoints(img::IntensityMap) = domainpoints(axisdims(img))


struct LazySlice{T, N, A<:AbstractVector{T}} <: AbstractArray{T, N}
    slice::A
    dir::Int
    dims::Dims{N}
    @inline function LazySlice(slice::AbstractVector{T}, dim::Int, dims::Dims{N}) where {T, N}
        @assert 1 ≤ dim ≤ N "Slice dimension is not valid. Must be ≤ $N and ≥ 1 and got $dim"
        return new{T, N, typeof(slice)}(slice, dim, dims)
    end
end

@inline Base.size(A::LazySlice) = A.dims
Base.@propagate_inbounds @inline function Base.getindex(A::LazySlice{T, N}, I::Vararg{Int, N}) where {T, N}
    i = I[A.dir]
    @boundscheck checkbounds(A.slice, i)
    return A.slice[i]
end

# The ntuple construction fails here (recursion limit?) so we use
# a generated function to get around it
@generated function _build_slices(g, sz::Dims{N}) where {N}
    exprs = [:(LazySlice(g[$k], $k, sz)) for k in 1:N]
    return :((tuple($(exprs...))))
end

# I need this because LazySlice is allocating unless I strip the dimension information
@inline basedim(x::DD.Dimension) = basedim(parent(x))
@inline basedim(x::DD.LookupArrays.LookupArray) = basedim(parent(x))
@inline basedim(x) = x




"""
    phasecenter(img::IntensityMap)

Computes the phase center of an intensity map. Note this is the pixels that is in
the middle of the image.
"""
function phasecenter(dims::AbstractRectiGrid)
    (;X, Y) = dims
    x0 = -(last(X) + first(X))/2
    y0 = -(last(Y) + first(Y))/2
    return (X=x0, Y=y0)
end
phasecenter(img::IntensityMap) = phasecenter(axisdims(img))



ChainRulesCore.@non_differentiable pixelsizes(img::IntensityMap)

"""
    imagepixels(fovx, fovy, nx, ny; x0=0, y0=0, executor=Serial(), header=NoHeader())

Construct a grid of pixels with a field of view `fovx` and `fovy` and `nx` and `ny` pixels.
This points are the pixel centers and the field of view goes from the edge of the first pixel
to the edge of the last pixel. The `x0`, `y0` offsets shift the image origin over by
(`x0`, `y0`) in the image plane.

## Arguments:
 - `fovx::Real`: The field of view in the x-direction
 - `fovy::Real`: The field of view in the y-direction
 - `nx::Integer`: The number of pixels in the x-direction
 - `ny::Integer`: The number of pixels in the y-direction

## Keyword Arguments:
 - `x0::Real=0`: The x-offset of the image
 - `y0::Real=0`: The y-offset of the image
 - `executor=Serial()`: The executor to use for the grid, default is serial execution
 - `header=NoHeader()`: The header to use for the grid
"""
function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, x0::Real = 0, y0::Real = 0; executor=Serial(), header=NoHeader())
    @assert (nx > 0)&&(ny > 0) "Number of pixels must be positive"

    psizex=fovx/nx
    psizey=fovy/ny

    xitr = X(LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2 - x0, nx))
    yitr = Y(LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2 - y0, ny))
    grid = RectiGrid((xitr, yitr); executor, header)
    return grid
end


"""
    fieldofview(img::IntensityMap)
    fieldofview(img::IntensityMap)

Returns a named tuple with the field of view of the image.
"""
function fieldofview(img::IntensityMap)
    return fieldofview(axisdims(img))
end

pixelsizes(img::IntensityMap) = pixelsizes(axisdims(img))



"""
    flux(im::IntensityMap)

Computes the flux of a intensity map
"""
function flux(im::IntensityMap{T,N}) where {T,N}
    return sum(im, dims=(:X, :Y))
end

flux(im::SpatialIntensityMap) = sum(parent(im))






"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.

For polarized maps we return the centroid for Stokes I only.
"""
function centroid(im::IntensityMap{<:Number})
    (;X, Y) = named_dims(im)
    return mapslices(x->centroid(IntensityMap(x, RectiGrid((;X, Y)))), im; dims=(:X, :Y))
end
centroid(im::IntensityMap{<:StokesParams}) = centroid(stokes(im, :I))

function centroid(im::IntensityMap{T,2})::Tuple{T,T} where {T<:Real}
    f = flux(im)
    cent = sum(pairs(DimPoints(im)); init=SVector(zero(f), zero(f))) do (I, (x, y))
        x0 = x.*im[I]
        y0 = y.*im[I]
        return SVector(x0, y0)
    end
    return cent[1]./f, cent[2]./f
end

function ChainRulesCore.rrule(::typeof(centroid), img::IntensityMap{T,2}) where {T<:Real}
    out = centroid(img)
    x0, y0 = out
    pr = ProjectTo(img)
    function _centroid_pullback(Δ)
        f = flux(img)
        Δf = NoTangent()
        Δimg = Δ[1].*(img.X./f .- x0/f) .+ Δ[2].*(img.Y'./f .- y0/f)
        return Δf, pr(Δimg)
    end
    return out, _centroid_pullback
end


"""
    second_moment(im::AbstractIntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.

For polarized maps we return the second moment for Stokes I only.
"""
function second_moment(im::IntensityMap{T,N}; center=true) where {T<:Real,N}
    (;X, Y) = named_dims(im)
    return mapslices(x->second_moment(IntensityMap(x, RectiGrid((;X, Y))); center), im; dims=(:X, :Y))
end

# Only return the second moment for Stokes I
second_moment(im::IntensityMap{<:StokesParams}; center=true) = second_moment(stokes(im, :I); center)


"""
    second_moment(im::AbstractIntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.
"""
function second_moment(im::IntensityMap{T,2}; center=true) where {T<:Real}
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    for (I, (x,y)) in pairs(DimPoints(im))
        xx += x.^2*im[I]
        yy += y.^2*im[I]
        xy += x.*y.*im[I]
    end

    if center
        x0, y0 = centroid(im)
        xx = xx./f - x0.^2
        yy = yy./f - y0.^2
        xy = xy./f - x0.*y0
    else
        xx = xx./f
        yy = yy./f
        xy = xy./f
    end

    return @SMatrix [xx xy; xy yy]
end
