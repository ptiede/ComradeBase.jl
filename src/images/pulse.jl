export DeltaPulse, SqExpPulse, BSplinePulse, BicubicPulse, RaisedCosinePulse

"""
Pulse
Pixel response function for a radio image model. This makes
a discrete sampling continuous by picking a certain *smoothing*
kernel for the image.

# Notes
To see the implemented Pulses please use the subtypes function i.e.
`subtypes(Pulse)`
"""
abstract type Kernel <: AbstractModel end

visanalytic(::Type{<:Kernel}) = IsAnalytic()
imanalytic(::Type{<:Kernel}) = IsAnalytic()
isprimitive(::Type{<:Kernel}) = IsPrimitive()

@inline intensity_point(k::Kernel, p) = κ(k, p[:X])*κ(k, p[:Y])
@inline visibility_point(k::Kernel, p) = ω(k, p[:U])*ω(k, p[:V])


flux(p::Kernel) = κflux(p)^2

"""
    $(TYPEDEF)
A dirac comb pulse function. This means the image is just the
dicrete Fourier transform
"""
struct DeltaKernel{T} <: Kernel end
DeltaKernel() = DeltaKernel{Float64}()
# This should really be a delta function but whatever
κflux(::DeltaKernel{T}) where {T} = one(T)
@inline κ(::DeltaKernel{T}, x) where {T} = abs(x) < 0.5 ? one(T) : zero(T)
@inline ω(::DeltaKernel{T}, u) where {T} = one(T)
@inline radialextent(::Kernel) = 1.0

"""
    $(TYPEDEF)
Normalized square exponential kernel, i.e. a Gaussian. Note the
smoothness is modfied with `ϵ` which is the inverse variance in units of
1/pixels².
"""
struct SqExpKernel{T} <: Kernel
    ϵ::T
end
@inline @fastmath κ(b::SqExpKernel, x) = exp(-0.5*b.ϵ^2*x^2)/sqrt(2*π/b.ϵ^2)
@inline κflux(::SqExpKernel{T}) where {T} = one(T)
@inline @fastmath ω(b::SqExpKernel, u) = exp(-2*(π*u/b.ϵ)^2)
@inline radialextent(p::SqExpKernel) = 5/p.ϵ

@doc raw"""
    $(TYPEDEF)
Uses the basis spline (BSpline) kernel of order `N`. These are the kernel that come
from recursively convolving the tophat kernel
```math
    B_0(x) = \begin{cases} 1 & |x| < 1 \\ 0 & otherwise \end{cases}
```
`N` times.

## Notes

BSpline kernels have a number of nice properties:
1. Simple frequency response $\sinc(u/2)^N$
2. preserve total intensity

For `N`>1 these kernels aren't actually interpolation kernels however, this doesn't matter
for us.

Currently only the 0,1,3 order kernels are implemented.
"""
struct BSplineKernel{N} <: Kernel end
@inline ω(::BSplineKernel{N}, u) where {N} = sinc(u)^(N+1)
@inline κflux(::BSplineKernel) = 1.0

@inline κ(::BSplineKernel{0}, x::T) where {T} = abs(x) < 0.5 ? one(T) : zero(T)

@inline function κ(::BSplineKernel{1}, x::T) where {T}
    mag = abs(x)
    return mag < 1 ? 1-mag : zero(T)
end

@inline function κ(::BSplineKernel{3}, x::T) where {T}
    mag = abs(x)
    if mag < 1
        return evalpoly(mag, (4, 0, -6, 3))/6
    elseif 1 ≤ mag < 2
        return evalpoly(mag, (8, -12, 6, -1))/6
    else
        return zero(T)
    end
end


struct BicubicPulse{T} <: Pulse
    b::T
end

BicubicPulse() = BicubicPulse{Float64}(-0.5)

function κ(k::BicubicPulse, x::T) where {T}
    mag = abs(x)
    b = k.b
    if mag < 1
        return evalpoly(mag, (one(T), zero(T), -(b+3), b+2))
    elseif 1 ≤ mag < 2
        return b*evalpoly(mag, (-T(4), T(8), -T(5), one(T)))
    else
        return zero(T)
    end
end

function ω(m::BicubicPulse, u)
    b = m.b
    k = 2π*u
    abs(k) < 1e-2 && return 1 - (2*b + 1)*k^2/15 + (16*b + 1)*k^4/560
    s,c = sincos(k)
    k3 = k^3
    k4 = k3*k
    c2 = c^2 - s^2
    return -4*s*(2*b*c + 4*b + 3)*inv(k3) + 12*inv(k4)*(b*(1-c2) + 2*(1-c))
end

struct RaisedCosinePulse{T} <: Pulse
    rolloff::T
end

RaisedCosinePulse() = RaisedCosinePulse{Float64}(0.5)

function κ(k::RaisedCosinePulse, x::T) where {T}
    mag = abs(x)
    β = k.rolloff
    if 2*mag < 1-β
        return one(T)
    elseif 1-β <= 2*mag <= 1+β
        return 1/2*(1 + cospi((mag - (1-β)/2)/β))
    else
        return zero(T)
    end
end

function ω(k::RaisedCosinePulse, u::T) where {T}
    β = k.rolloff
    return sinc(u)*cos(β*u)*inv(1 - (2β*u/π)^2)
end
