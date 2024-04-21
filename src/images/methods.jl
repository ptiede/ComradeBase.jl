"""
    domaingrid(k::IntensityMap)

Returns the grid the `IntensityMap` is defined as. Note that this is unallocating
since it lazily computes the grid. The grid is an example of a DimArray and works similarly.
This is useful for broadcasting a model across an abritrary grid.
"""
domaingrid(img::IntensityMapTypes) = domaingrid(axisdims(img))


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
    phasecenter(img::StokesIntensitymap)

Computes the phase center of an intensity map. Note this is the pixels that is in
the middle of the image.
"""
function phasecenter(dims::AbstractGrid)
    (;X, Y) = dims
    x0 = -(last(X) + first(X))/2
    y0 = -(last(Y) + first(Y))/2
    return (X=x0, Y=y0)
end
phasecenter(img::IntensityMapTypes) = phasecenter(axisdims(img))


"""
    imagepixels(img::IntensityMap)
    imagepixels(img::IntensityMapTypes)

Returns a abstract spatial dimension with the image pixels locations `X` and `Y`.
"""
imagepixels(img::IntensityMapTypes) = (X=img.X, Y=img.Y)

ChainRulesCore.@non_differentiable imagepixels(img::IntensityMapTypes)
ChainRulesCore.@non_differentiable pixelsizes(img::IntensityMapTypes)

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
    fieldofview(img::IntensityMapTypes)

Returns a named tuple with the field of view of the image.
"""
function fieldofview(img::IntensityMapTypes)
    return fieldofview(axisdims(img))
end

function fieldofview(dims::RectiGrid)
    (;X,Y) = dims
    dx = step(X)
    dy = step(Y)
    (X=abs(last(X) - first(X))+dx, Y=abs(last(Y)-first(Y))+dy)
end


"""
    pixelsizes(img::IntensityMap)
    pixelsizes(img::AbstractGrid)

Returns a named tuple with the spatial pixel sizes of the image.
"""
function pixelsizes(keys::AbstractGrid)
    x = keys.X
    y = keys.Y
    return (X=step(x), Y=step(y))
end
pixelsizes(img::IntensityMapTypes) = pixelsizes(axisdims(img))



"""
    flux(im::IntensityMap)
    flux(img::StokesIntensityMap)

Computes the flux of a intensity map
"""
function flux(im::IntensityMapTypes{T,N}) where {T,N}
    return sum(im, dims=(:X, :Y))
end

flux(im::SpatialIntensityMap) = sum(im)
flux(img::StokesIntensityMap{T}) where {T} = StokesParams(flux(stokes(img, :I)),
                                             flux(stokes(img, :Q)),
                                             flux(stokes(img, :U)),
                                             flux(stokes(img, :V)),
                                             )






"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.

For polarized maps we return the centroid for Stokes I only.
"""
function centroid(im::IntensityMapTypes{<:Number})
    (;X, Y) = named_axisdims(im)
    return mapslices(x->centroid(IntensityMap(x, (;X, Y))), im; dims=(:X, :Y))
end
centroid(im::IntensityMapTypes{<:StokesParams}) = centroid(stokes(im, :I))

function centroid(im::IntensityMapTypes{T,2})::Tuple{T,T} where {T<:Real}
    f = flux(im)
    cent = sum(pairs(DimPoints(im)); init=SVector(zero(f), zero(f))) do (I, (x, y))
        x0 = x.*im[I]
        y0 = y.*im[I]
        return SVector(x0, y0)
    end
    return cent[1]./f, cent[2]./f
end

function ChainRulesCore.rrule(::typeof(centroid), img::IntensityMapTypes{T,2}) where {T<:Real}
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
function second_moment(im::IntensityMapTypes{T,N}; center=true) where {T<:Real,N}
    (;X, Y) = named_axisdims(im)
    return mapslices(x->second_moment(IntensityMap(x, (;X, Y)); center), im; dims=(:X, :Y))
end

# Only return the second moment for Stokes I
second_moment(im::IntensityMapTypes{<:StokesParams}; center=true) = second_moment(stokes(im, :I); center)


"""
    second_moment(im::AbstractIntensityMap; center=true)

Computes the image second moment tensor of the image.
By default we really return the second **cumulant** or centered
second moment, which is specified by the `center` argument.
"""
function second_moment(im::IntensityMapTypes{T,2}; center=true) where {T<:Real}
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
