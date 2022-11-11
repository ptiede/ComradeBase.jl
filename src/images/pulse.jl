

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
