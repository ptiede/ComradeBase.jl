abstract type AbstractIntensityMap{T,S} <: AbstractMatrix{T} end


#abstract type AbstractPolarizedMap{I,Q,U,V} end

@inline function intensitymap(s::M,
                              fovx::Number, fovy::Number,
                              nx::Int, ny::Int;
                              pulse=ComradeBase.DeltaPulse()) where {M}
    return intensitymap(imanalytic(M), s, fovx, fovy, nx, ny; pulse)
end

@inline function intensitymap!(img::AbstractIntensityMap, s::M) where {M}
    return intensitymap!(imanalytic(M), img, s)
end

function intensitymap(::IsAnalytic, s, fovx::Number, fovy::Number, nx::Int, ny::Int; pulse=ComradeBase.DeltaPulse())
    x,y = imagepixels(fovx, fovy, nx, ny)
    pimg = map(CartesianIndices((1:ny,1:nx))) do I
        iy,ix = Tuple(I)
        @inbounds f = intensity_point(s, x[ix], y[iy])
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



include("pulse.jl")
include("polarizedtypes.jl")
include("intensitymap.jl")
#include("polarizedmap.jl")
