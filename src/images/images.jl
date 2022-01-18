abstract type AbstractIntensityMap{T,S} <: AbstractMatrix{T} end

#abstract type AbstractPolarizedMap{I,Q,U,V} end

@inline function intensitymap(s::M,
                              fovx::Number, fovy::Number,
                              nx::Int, ny::Int;
                              pulse=ComradeBase.DeltaPulse()) where {M}
    return intensitymap(imanalytic(M), s, fovx, fovy, nx, ny; pulse)
end

function intensitymap(::IsAnalytic, s, fovx::Number, fovy::Number, nx::Int, ny::Int; pulse=ComradeBase.DeltaPulse())
    px = fovx/max(nx-1,1)
    py = fovy/max(ny-1,1)
    pimg = map(CartesianIndices((1:nx,1:nx))) do I
        iy,ix = Tuple(I)
        x = -fovx/2 + (ix-1)*px
        y = -fovy/2 + (iy-1)*py
        f = intensity_point(s, x, y)
        return f
    end
    return IntensityMap(pimg, fovx, fovy, pulse)
end

function intensitymap!(::IsAnalytic, im::AbstractIntensityMap, m)
    xitr, yitr = imagepixels(im)
    @inbounds for (i,x) in pairs(xitr), (j,y) in pairs(yitr)
        im[j, i] = intensity_point(m, x, y)
    end
    return im
end


"""
Pulse
Pixel response function for a radio image model. This makes
a discrete sampling continuous by picking a certain *smoothing*
kernel for the image.

# Notes
To see the implemented Pulses please use the subtypes function i.e.
`subtypes(Pulse)`
"""
abstract type Pulse <: AbstractModel end

visanalytic(::Type{<:Pulse}) = IsAnalytic()
imanalytic(::Type{<:Pulse}) = IsAnalytic()
isprimitive(::Type{<:Pulse}) = IsPrimitive()

@inline intensity_point(p::Pulse, x,y) = κ(p::Pulse, x)*κ(p::Pulse, y)
@inline visibility_point(p::Pulse, u,v) = ω(p::Pulse, u)*ω(p::Pulse, u)

include("pulse.jl")
include("polarizedtypes.jl")
include("intensitymap.jl")
#include("polarizedmap.jl")
