export fov, imagepixels, pixelsizes, stokes, centroid, inertia, phasecenter

"""
    ContinuousImage{A<:AbstractDimArray, P}


"""
struct ContinuousImage{A <: AbstractDimArray, P} <: AbstractModel
    """
    Discrete representation of the image. This must be a DimArray where at least two of the
    """
    img::A
    """
    Image Kernel that transforms from the discrete image to a continuous one. This is
    sometimes called a pulse function.
    """
    kernel::P
end

function ContinuousImage(im::AbstractMatrix, fovx::Real, fovy::Real, x0::Real, y0::Real, pulse; kwargs...)
    xitr, yitr = imagepixels(fovx, fovy, x0, y0, size(im,2), size(im,2))
    img = DimArray(im, (X=xitr, Y=yitr); kwargs...)
    return ContinuousImage(img, pulse)
end

function ContinuousImage(im::AbstractMatrix, fov::Real, x0::Real, y0::Real, pulse; kwargs...)
    return ContinuousImage(im, fov, fov, x0, y0, pulse; kwargs...)
end

Base.getindex(img::ContinuousImage, args...) = getindex(img.img, args...)
Base.setindex!(img::ContinuousImage, args...) = setindex!(img.img, args...)


imagepixels(img::ContinuousImage) = NamedTuple{names.(dims(img.img))}(dims(img.img))

# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()
isprimitive(::Type{<:ContinuousImage}) = IsPrimitive()

function intensity_point(m::ContinuousImage, p)
    x = p[:X]
    y = p[:Y]
    dx, dy = pixelsizes(m)
    xitr,yitr = imagepixels(m)
    sum = zero(eltype(m))
    #The kernel is written in terms of pixel number so we convert x to it
    @inbounds for p in DimPoints(m.img)
        k = intensity_point(m.pulse, p)
        #println(Δx," ", Δy, " ",   k)
        sum += m[j, i]*k/(dx*dy)
    end
    return sum
end



"""
    flux(im::AbstractDimArray)

Computes the flux of a intensity map
"""
function flux(im::AbstractDimArray)
    f = sum(im, dims=(:X, :Y))#*(flux(im.pulse))^2
    return f#*prod(pixelsizes(im))
end


const SpatialOnly = Union{Tuple{<:X, <:Y}, Tuple{<:Y, <:X}}


"""
    centroid(im::AbstractIntensityMap)

Computes the image centroid aka the center of light of the image.
"""
function centroid(im::DimArray{T,2, <:SpatialOnly}) where {T}
    xitr, yitr = imagepixels(im)
    x0 = zero(T)
    y0 = zero(T)
    f = flux(im)
    for i in axes(im,:X), j in axes(im,:Y)
        x0 += xitr[i].*im[X=i, Y=j]
        y0 += yitr[j].*im[X=I, Y=j]
    end
    return x0/f, y0/f
end

"""
    inertia(im::AbstractIntensityMap; center=true)

Computes the image inertia aka **second moment** of the image.
By default we really return the second **cumulant** or second centered
second moment, which is specified by the `center` argument.
"""
function inertia(im::DimArray{T,2, <:SpatialOnly}; center=true) where {T}
    xx = zero(T)
    xy = zero(T)
    yy = zero(T)
    f = flux(im)
    xitr, yitr = imagepixels(im)
    for i in axes(img, :X) j in axes(img, :Y)
        x = xitr[i]
        y = yitr[j]
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




# Define the array interface
# Base.IndexStyle(::Type{<: AbstractIntensityMap{T,S}}) where {T,S} = Base.IndexStyle(S)
# Base.size(im::AbstractIntensityMap) = size(im.img)
# Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, i::Int) = getindex(im.img, i)
# Base.@propagate_inbounds Base.getindex(im::AbstractIntensityMap, I...) = getindex(im.img, I...)
# Base.@propagate_inbounds Base.setindex!(im::AbstractIntensityMap, x, i::Int) = setindex!(im.img, x, i)
# Base.@propagate_inbounds Base.setindex!(im::AbstractIntensityMap, x, i) = setindex!(im.img, x, i)

# function Base.similar(im::IntensityMap, ::Type{T}) where{T}
#     sim = similar(im.img, T)
#     return IntensityMap(sim, fov(im), im.phasecenter, im.pulse)
# end

# #function Base.similar(im::AbstractIntensityMap, ::Type{T}, dims::Dims) where {T}
# #    fovx = im.psizex*last(dims)
# #    fovy = im.psizey*first(dims)
# #    sim = similar(im.im, T, dims)
# #    return IntensityMap(sim, fovx, fovy, im.psizex, im.psizey)
# #end

# #Define the broadcast interface
# struct IntensityMapStyle <: Broadcast.AbstractArrayStyle{2} end
# IntensityMapStyle(::Val{2}) = IntensityMapStyle()

# Base.BroadcastStyle(::Type{<:AbstractIntensityMap}) = IntensityMapStyle()
# function Base.similar(bc::Broadcast.Broadcasted{IntensityMapStyle}, ::Type{ElType}) where ElType
#     #Scan inputs for IntensityMap
#     #print(bc.args)
#     Im = _find_sim(bc)
#     #fovxs = getproperty.(Im, Ref(:fovx))
#     #fovys = getproperty.(Im, Ref(:fovy))
#     #@assert all(i->i==first(fovxs), fovxs) "IntensityMap fov must be equal to add"
#     #@assert all(i->i==first(fovys), fovys) "IntensityMap fov must be equal to add"
#     return IntensityMap(similar(Array{ElType}, axes(bc)), fov(Im), Im.phasecenter, Im.pulse)
# end

# #Finds the first IntensityMap and uses that as the base
# #TODO: If multiply IntensityMaps maybe I should make sure they are consistent?
# _find_sim(bc::Base.Broadcast.Broadcasted) = _find_sim(bc.args)
# _find_sim(args::Tuple) = _find_sim(_find_sim(args[1]), Base.tail(args))
# _find_sim(x) = x
# _find_sim(::Tuple{}) = nothing
# _find_sim(a::AbstractIntensityMap, _) = a
# _find_sim(::Any, rest) = _find_sim(rest)

# #Guards to prevent someone from adding two Images with different FOV's
# function Base.:+(x::AbstractIntensityMap, y::AbstractIntensityMap)
#     @assert fov(x) == fov(y) "IntensityMaps must share same field of view"
#     @assert phasecenter(x) == phasecenter(y) "IntensityMaps must share same phasecenter"
#     return x .+ y
# end


# @inline function pixelsizes(im::AbstractIntensityMap)
#     ny,nx = size(im)
#     fovx, fovy = fov(im)
#     return fovx/nx, fovy/ny
# end

# @inline function imagepixels(fovx, fovy, x0, y0, nx::Int, ny::Int)
#     px = fovx/nx; py = fovy/ny
#     return range(-fovx/2+px/2 - x0, step=px, length=nx),
#            range(-fovy/2+py/2 - y0, step=py, length=ny)
# end

# @inline function imagepixels(im::AbstractIntensityMap)
#     ny,nx = size(im)
#     x0 ,y0 = phasecenter(im)
#     fovx, fovy = fov(im)
#     return imagepixels(fovx, fovy, x0, y0, nx, ny)
# end
